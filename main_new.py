import os, re
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rich import print
load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", output_dimensionality=768)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name is UNIQUE")

index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

neo4j_count = graph.query("MATCH (m:Movie) Return count(m) AS total")[0]["total"]
pinecone_stats = pc.Index(index_name).describe_index_stats()
vector_count = pinecone_stats.total_vector_count
LIMIT = 1000
if neo4j_count > 0 and vector_count > 0:
    print(f"Detected {neo4j_count} Neo4j movies and {vector_count} Pinecone vectors. Skipping ingestion!")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
else:
    print("Databases are empty! Starting one-time ingestion...")
    loader = CSVLoader(
        file_path="wiki_movie_plots_deduped.csv",
        encoding="utf-8", 
        metadata_columns=["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page"],
        content_columns=["Plot"],
    )
    docs = loader.load()

    cypher_query = """
    MERGE (m:Movie {title:$title})
    SET m.releaseYear = $year,
        m.origin = $origin,
        m.wikiPage = $wikiPage
    MERGE (g:Genre {name:$genre})
    MERGE (m)-[:HAS_GENRE]->(g)
    FOREACH ( dName In $directors | MERGE (d:Person {name :dName}) MERGE (d)-[:DIRECTED]->(m))
    FOREACH ( aName In $actors    | MERGE (a:Person {name:aName}) MERGE (a)-[:ACTED_IN]->(m))
    """
    for doc in docs[:LIMIT]:
        meta = doc.metadata

        # Filter out empty names and 'Unknown'
        directors = [d.strip() for d in meta.get("Director", "").split(",") if d.strip() and d.lower() != "unknown"]
        actors = [a.strip() for a in meta.get("Cast", "").split(",") if a.strip() and a.lower() != "unknown"]

        params = { 
            "title": meta.get("Title", "Unknown"),
            "year": int(meta["Release Year"]) if meta.get("Release Year", "").isdigit() else None,
            "origin": meta.get("Origin/Ethnicity", "Unknown"),
            "wikiPage": meta.get("Wiki Page", ""),
            "genre": meta.get("Genre", "Unknown").strip().lower() or "unknown",
            "directors": directors,
            "actors": actors
        }
        graph.query(cypher_query, params=params)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    plot_chunks = text_splitter.split_documents(docs[:LIMIT])
    vector_store = PineconeVectorStore.from_documents(documents=plot_chunks, embedding=embeddings, index_name=index_name)
    print("One-time ingestion complete!")


@tool
def vector_tool(query: str, k: int = 3) -> str:
    """
    Search the Pinecone vector database for movie plot summaries, scene descriptions, themes, and narrative details.
    Use this when the user asks about what happens in a story, specific actions, or thematic queries.
    """
    print("Vector Tool Calling ............")
    docs = vector_store.similarity_search(query, k)
    if not docs:
        return "No Relevent movie plot chunks plot"
    formatted_results = []
    for i in range(len(docs)):
        doc = docs[i]
        title = doc.metadata.get("Title", "Unknown")
        genre = doc.metadata.get("Genre", "Unknown")
        year = doc.metadata.get("Release Year", "Unknown")
        page_content = doc.page_content
        formatted_results.append(
            f"[Result {i}]\nMovie: {title} ({year})\nGenre: {genre}\nPlot: {page_content}\n"
        )
    print(formatted_results)
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
3. Use case-insensitive matching where applicable: WHERE toLower(m.title) CONTAINS toLower('...')
FAILED QUERY:
{failed_query}
ERROR MESSAGE FROM NEO4J:
{error_message}
CORRECTED CYPHER QUERY:"""

model = init_chat_model("google_genai:gemini-2.5-flash")
strParser = StrOutputParser()
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
    - Use case-insensitive matching: WHERE toLower(m.title) CONTAINS toLower('...')
    - Return specific columns using AS aliases (e.g., RETURN p.name AS Director, m.title AS Movie)
    """
    print("Graph Tool Calling ............")
    current_query = cypher_query
    max_retries = 3
    for attempt in range(max_retries + 1):
        clean_cypher = extract_cypher(current_query)
        is_safe, securtiy_msg = check_cypher_guardrail(clean_cypher)
        if not is_safe:
            print(f"GaurdRail Blocked :{securtiy_msg}")
            if attempt < max_retries:
                corrected = correction_chain.invoke({
                    "failed_query": clean_cypher,
                    "error_message": securtiy_msg
                })
                current_query = corrected
                continue
            return securtiy_msg
               
        try:
            print(f"Graph Tool [Attempt {attempt+1}]:Executing -> {clean_cypher}")
            results = graph.query(clean_cypher)
            if results:
                print(type(results)) 
                print(results)
                return str(results)    
            return "No graph records found for this query."
               
        except Exception as e:
            error_msg = str(e)
            print(f" Neo4j Error on Attempt {attempt + 1}: {error_msg}")
            if attempt < max_retries:
                print("Triggering Self-Healing LLM to fix syntax...")
                corrected = correction_chain.invoke({
                    "failed_query": clean_cypher,
                    "error_message": error_msg
                })
                current_query = corrected
            else:
                return f"Cypher Execution Failed after retries: {error_msg}"
            
    return "Unable to execute Query"

tools = [vector_tool, graph_tool]

tools_with_name = {t.name: t for t in tools}
llm_with_tools = model.bind_tools(tools)

system_prompt_text = """You are an advanced Movie Intelligence Assistant powered by a Hybrid GraphRAG architecture.

You have access to two specialized tools:
1. `graph_tool`:
   - Use for structured, relational, and factual questions (e.g., directors, cast members, release years, genres, cross-movie connections).
   - Generate precise, read-only Cypher queries following the graph schema.
2. `vector_tool`:
   - Use for semantic, thematic, and descriptive questions (e.g., plot summaries, specific scenes, storylines, emotions, narrative themes).

CORE GUIDELINES:
- **Parallel Tool Calling**: If a user's question asks for both factual data (director, year, genre) AND story details, call BOTH `graph_tool` and `vector_tool` together in the same turn.
- **Handling Missing Data / 0 Records**: If tools return no records or no relevant plot chunks, politely inform the user. Never hallucinate or invent facts.
- **Synthesize & Format**: Always summarize raw tool outputs into clear, professional, and well-structured markdown (using bullet points and bold highlights). Never leave your final response empty.
- **Conversational Memory**: Maintain full context across multi-turn questions."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="messages")
])

chain_with_tools = prompt_template | llm_with_tools
chain_without_tools = prompt_template | model

message_history = []

def chat(user_input: str) -> str:
    message_history.append(HumanMessage(content=user_input))

    ai_response = chain_with_tools.invoke({"messages": message_history})
    message_history.append(ai_response)

    if ai_response.tool_calls:
        print(f"Tool calling {ai_response.tool_calls}")
        for tool_call in ai_response.tool_calls:
            selected_tool = tools_with_name[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])

            message_history.append(
                ToolMessage(content=str(tool_output),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                            )
            )
        text = chain_without_tools.invoke({"messages": message_history})
        message_history.append(text)
        return strParser.invoke(text)   

    return strParser.invoke(ai_response)

while True:
    user_query = input("You: ").strip()
    if user_query.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!\n")
            break
    if not user_query:
            continue
    response = chat(user_query)
    print(f"\nAI: {response}\n")
