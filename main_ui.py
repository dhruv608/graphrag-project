import os
import re
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Page Configuration
st.set_page_config(
    page_title="Movie Assistant",
    page_icon="🎬",
    layout="centered"
)

load_dotenv()

# 2. Cached Backend Initialization (Runs Once)
@st.cache_resource(show_spinner="Initializing Database & Models...")
def init_backend():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768
    )

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE")
    )

    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")

    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    neo4j_count = graph.query("MATCH (m:Movie) RETURN count(m) AS total")[0]["total"]
    pinecone_stats = pc.Index(index_name).describe_index_stats()
    vector_count = pinecone_stats.total_vector_count

    LIMIT = 1000
    if neo4j_count > 0 and vector_count > 0:
        vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    else:
        loader = CSVLoader(
            file_path="wiki_movie_plots_deduped.csv",
            encoding="utf-8",
            metadata_columns=["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page"],
            content_columns=["Plot"],
        )
        docs = loader.load()
        csvParser = CommaSeparatedListOutputParser()

        cypher_query = """
        MERGE (m:Movie {title:$title})
        SET m.releaseYear = $year,
        m.origin = $origin,
        m.wikiPage = $wikiPage,
        m.plot = $plot
        MERGE (g:Genre {name:$genre})
        MERGE (m)-[:HAS_GENRE]->(g)
        FOREACH ( dName IN $directors | MERGE (d:Person {name :dName}) MERGE (d)-[:DIRECTED]->(m))
        FOREACH ( aName IN $actors    | MERGE (a:Person {name:aName}) MERGE (a)-[:ACTED_IN]->(m))
        """

        for doc in docs[:LIMIT]:
            meta = doc.metadata
            params = {
                "title": meta.get("Title", "Unknown"),
                "year": int(meta["Release Year"]) if meta.get("Release Year", "").isdigit() else None,
                "origin": meta.get("Origin/Ethnicity", "Unknown"),
                "wikiPage": meta.get("Wiki Page", ""),
                "plot": doc.page_content,
                "genre": meta.get("Genre", "Unknown").strip().lower() or "unknown",
                "directors": [d for d in csvParser.parse(meta.get("Director", "")) if d.lower() != "unknown"],
                "actors": [a for a in csvParser.parse(meta.get("Cast", "")) if a.lower() != "unknown"]
            }
            graph.query(cypher_query, params=params)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        plot_chunks = text_splitter.split_documents(docs[:LIMIT])
        vector_store = PineconeVectorStore.from_documents(
            documents=plot_chunks,
            embedding=embeddings,
            index_name=index_name
        )

    model = init_chat_model("google_genai:gemini-2.5-flash")
    strParser = StrOutputParser()

    return embeddings, graph, vector_store, model, strParser

embeddings, graph, vector_store, model, strParser = init_backend()

# 3. Tools
@tool
def vector_tool(query: str, k: int = 3) -> str:
    """
    Search the Pinecone vector database for movie plot summaries, scene descriptions, themes, and narrative details.
    Use this when the user asks about what happens in a story, specific actions, or thematic queries.
    """
    docs = vector_store.similarity_search(query, k)
    if not docs:
        return "No relevant movie plot chunks found."
    formatted_results = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("Title", "Unknown")
        genre = doc.metadata.get("Genre", "Unknown")
        year = doc.metadata.get("Release Year", "Unknown")
        page_content = doc.page_content
        formatted_results.append(
            f"[Result {i + 1}]\nMovie: {title} ({year})\nGenre: {genre}\nPlot: {page_content}\n"
        )
    return "\n\n".join(formatted_results)

FORBIDDEN_WORDS = ["DELETE", "DETACH", "DROP", "CREATE", "SET", "REMOVE", "MERGE"]

def check_cypher_guardrail(query: str) -> tuple[bool, str]:
    """Ensures query is read-only and blocks destructive operations"""
    upper_query = query.upper()
    for word in FORBIDDEN_WORDS:
        if word in upper_query:
            return False, f"Security Violation: Command '{word}' is blocked!"
    if "MATCH" not in upper_query or "RETURN" not in upper_query:
        return False, "Security Violation: Query must be a read-only MATCH...RETURN statement."
    return True, "Valid"

correction_prompt = """You are a Neo4j Cypher expert.
A previously generated Cypher query failed. Analyze the error message and graph schema, and fix the query.
GRAPH SCHEMA:
- (:Person {{name}})-[:DIRECTED]->(:Movie {{title, releaseYear, origin, wikiPage}})
- (:Person {{name}})-[:ACTED_IN]->(:Movie {{title, releaseYear, origin, wikiPage}})
- (:Movie {{title}})-[:HAS_GENRE]->(:Genre {{name}})
RULES:
1. ONLY return the valid Cypher query inside a ```cypher ... ``` code block.
2. Ensure read-only syntax (MATCH and RETURN only).
3. Use case-insensitive matching where applicable: WHERE toLower(m.title) CONTAINS toLower('...') OR toLower(g.name) CONTAINS toLower('...')
FAILED QUERY:
{failed_query}
ERROR MESSAGE FROM NEO4J:
{error_message}
CORRECTED CYPHER QUERY:"""

correction_chain = ChatPromptTemplate.from_template(correction_prompt) | model | strParser

@tool
def graph_tool(cypher_query: str) -> str:
    """
    Execute a Cypher query against the Neo4j Movie Knowledge Graph.
    Schema:
    - (:Person {name})-[:DIRECTED]->(:Movie {title, releaseYear, origin, wikiPage})
    - (:Person {name})-[:ACTED_IN]->(:Movie {title, releaseYear, origin, wikiPage})
    - (:Movie {title})-[:HAS_GENRE]->(:Genre {name})
    
    Rules:
    - Use case-insensitive matching with toLower() CONTAINS for titles and genres (e.g. WHERE toLower(g.name) CONTAINS 'romance')
    - Return specific columns using AS aliases (e.g., RETURN p.name AS Director, m.title AS Movie, m.releaseYear AS Year, g.name AS Genre)
    """
    current_query = cypher_query
    max_retries = 2
    for attempt in range(max_retries + 1):
        clean_cypher = extract_cypher(current_query)
        is_safe, security_msg = check_cypher_guardrail(clean_cypher)
        if not is_safe:
            if attempt < max_retries:
                corrected = correction_chain.invoke({
                    "failed_query": clean_cypher,
                    "error_message": security_msg
                })
                current_query = corrected
                continue
            return security_msg
               
        try:
            results = graph.query(clean_cypher)
            if results:
                return str(results)    
            return "No graph records found for this query."
               
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries:
                corrected = correction_chain.invoke({
                    "failed_query": clean_cypher,
                    "error_message": error_msg
                })
                current_query = corrected
            else:
                return f"Cypher Execution Failed after retries: {error_msg}"
            
    return "Unable to execute query."

tools = [vector_tool, graph_tool]
tools_with_name = {t.name: t for t in tools}
llm_with_tools = model.bind_tools(tools)

system_prompt_text = """You are an advanced Movie Intelligence Assistant powered by a Hybrid GraphRAG architecture.

You have access to two specialized tools:
1. `graph_tool`:
   - Use for structured, relational, and factual questions (e.g., directors, cast members, release years, genres, cross-movie connections).
   - Generate precise, read-only Cypher queries following the graph schema.
   - For genres, always use fuzzy matching: `WHERE toLower(g.name) CONTAINS 'romance'` or `'action'`.
2. `vector_tool`:
   - Use for semantic, thematic, and descriptive questions (e.g., plot summaries, specific scenes, storylines, emotions, narrative themes).

CORE GUIDELINES:
- **Parallel Tool Calling**: If a user's question asks for both factual data (director, year, genre) AND story details, call BOTH `graph_tool` and `vector_tool` together.
- **Handling Missing Data / 0 Records**: If tools return no records or no relevant plot chunks, politely inform the user. Never hallucinate or invent facts.
- **Synthesize & Format**: Always summarize raw tool outputs into clear, professional, and well-structured markdown (using bullet points and bold highlights). Never leave your final response empty.
- **Conversational Memory**: Maintain full context across multi-turn questions."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="messages")
])

chain_with_tools = prompt_template | llm_with_tools
chain_without_tools = prompt_template | model

# 4. Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# 5. Clean Sidebar (Only Relevant Example Queries & Reset)
with st.sidebar:
    st.markdown("### 💡 Example Questions")
    
    examples = [
        "Who directed The Great Train Robbery and in what year?",
        "List romance movies from 1929 or 1930.",
        "Search for a movie plot where bandits rob a train on horseback.",
        "Who directed Show Boat (1929) and what is the storyline?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.preset_prompt = ex

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_history = []
        st.rerun()

# 6. Main Chat Header
st.title("🎬 Movie Intelligence Assistant")
st.caption("Ask questions about movies, directors, release years, genres, or search plot storylines.")

# Display past chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tools" in msg and msg["tools"]:
            with st.expander("🛠️ View Database Queries & Results", expanded=False):
                for t in msg["tools"]:
                    st.markdown(f"**Tool Used:** `{t['name']}`")
                    st.code(t["output"], language="json" if "graph" in t["name"] else "text")

# 7. User Input Handling
prompt = st.chat_input("Ask a question about movies...")
if "preset_prompt" in st.session_state:
    prompt = st.session_state.pop("preset_prompt")

if prompt:
    # Append & display User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.message_history.append(HumanMessage(content=prompt))

    # Run Agent & Assistant Response
    with st.chat_message("assistant"):
        tool_logs = []
        with st.spinner("Searching knowledge base..."):
            ai_response = chain_with_tools.invoke({"messages": st.session_state.message_history})
            st.session_state.message_history.append(ai_response)

            if ai_response.tool_calls:
                with st.status("⚡ Retrieving from Knowledge Graph & Vectors...", expanded=False) as status:
                    for tool_call in ai_response.tool_calls:
                        t_name = tool_call["name"]
                        t_args = tool_call["args"]
                        status.write(f"Querying `{t_name}`: `{t_args}`")
                        
                        selected_tool = tools_with_name[t_name]
                        tool_output = selected_tool.invoke(t_args)

                        tool_logs.append({
                            "name": t_name,
                            "output": str(tool_output)
                        })

                        st.session_state.message_history.append(
                            ToolMessage(
                                content=str(tool_output),
                                name=t_name,
                                tool_call_id=tool_call["id"]
                            )
                        )
                    status.update(label="Data retrieved", state="complete")

                text = chain_without_tools.invoke({"messages": st.session_state.message_history})
                st.session_state.message_history.append(text)
                final_answer = strParser.invoke(text)
            else:
                final_answer = strParser.invoke(ai_response)

        # Render final markdown answer
        st.markdown(final_answer)

        # Store to UI message state
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "tools": tool_logs
        })
