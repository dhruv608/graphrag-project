from dotenv import load_dotenv
load_dotenv()
from rich import print
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from pinecone import Pinecone
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.messages import HumanMessage , ToolMessage , AIMessage
from langchain_core.output_parsers import StrOutputParser
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")

neo4j_count = graph.query("MATCH (m:Movie) RETURN count(m) AS total")[0]["total"]
# [{'total': 0}]
index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vector_count = pc.Index(index_name).describe_index_stats().total_vector_count

limit = 1000
if neo4j_count>0 and vector_count>0:
    print(f"Already Data is there v->{vector_count}  g ->{neo4j_count}")

    vector_store = PineconeVectorStore(index_name=index_name , embedding=embeddings)
else:
    print("First time loading")
    loader = CSVLoader(
        file_path="wiki_movie_plots_deduped.csv",
        encoding="utf-8",
        content_columns=["Plot"],
        metadata_columns=["Release Year","Title","Origin/Ethnicity","Director","Cast","Genre","Wiki Page"]
        )
    docs = loader.load()
    hindi_docs = [doc for doc in docs if doc.metadata.get("Origin/Ethnicity") == "Bollywood"]
    print(hindi_docs[-5])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400 , chunk_overlap=50)
    plot_chunks = text_splitter.split_documents(hindi_docs[:limit])
    vector_store = PineconeVectorStore.from_documents(documents=plot_chunks,embedding=embeddings,index_name=index_name)
    # graph db:
    print("Vector db loading done")
    cypher_query = """
    MERGE (m:Movie {title:$title})
    SET m.releaseYear = $year,
        m.origin = $origin,
        m.wikiPage = $wikiPage
    MERGE (g:Genre {name:$genre})
    MERGE (m)-[:HAS_GENRE]->(g)
    FOREACH ( dName IN $directors | MERGE (d:Person {name:dName}) MERGE (d)-[:DIRECTED]->(m))
    FOREACH ( aName IN $actors    | MERGE (a:Person {name:aName}) MERGE (a)-[:ACTED_IN]->(m))
    """
    # "a,b,c"
    print("Graph db starting")
    
    for doc in hindi_docs[:limit]:
        meta=doc.metadata
        
        directors = [d.strip() for d in meta.get("Director","").split(",") if d.strip() and d.lower() !=  "unknown"]
        actors = [a.strip() for a in meta.get("Cast","").split(",") if a.strip() and a.lower() !=  "unknown"]
        params ={
            "title":meta.get("Title","Unknown"),
            "year": int(meta["Release Year"]) if meta.get("Release Year", "").isdigit() else None,
            "origin": meta.get("Origin/Ethnicity", "Unknown"),
            "wikiPage": meta.get("Wiki Page", ""),
            "genre": meta.get("Genre", "Unknown").strip().lower() or "unknown",
            "directors": directors,
            "actors": actors
        }
        graph.query(cypher_query,params=params)
    print("all database dumped full")
    

# vector tool and graph _tool
@tool
def vector_tool(query: str, k: int = 5) -> str:
    """
    Search the Pinecone vector database for movie plot summaries, scene descriptions, themes, and narrative details.
    Use this when the user asks about what happens in a story, specific actions, or thematic queries.
    """
    print("Vector Tool Calling ............")
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
            f"[Result {i}]\nMovie: {title} ({year})\nGenre: {genre}\nPlot: {page_content}\n"
        )
    print(formatted_results)
    return "\n\n".join(formatted_results)
# graph tool

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
    print("Graph Tool Calling .....")
    clean_cypher = extract_cypher(cypher_query)
    try:
        print(f"Executing Cypher -> {clean_cypher}")
        results = graph.query(clean_cypher)
        if results:
            print(results)
            return str(results)
        return "No graph records found for this query."
    except Exception as e:
        print(f"Neo4j Error: {e}")
        return f"Error executing Cypher query: {e}"


model = init_chat_model("google_genai:gemini-3.5-flash")
strParser = StrOutputParser()


tools = [vector_tool , graph_tool]
tools_with_name = {t.name :t for t in tools}
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
- **Handling Missing Data / 0 Records**: If tools return no records or no relevant plot chunks, politely inform the user. 
- Never hallucinate or invent facts.
- **Synthesize & Format**: Always summarize raw tool outputs into clear, professional, 
- and well-structured markdown (using bullet points and bold highlights). Never leave your final response empty.
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
                ToolMessage(
                    content=str(tool_output),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        text = chain_without_tools.invoke({"messages": message_history})
        message_history.append(text)
        return strParser.invoke(text)   
    
    return strParser.invoke(ai_response)

print("welcome to CHATBOT")
while True:
    user_query = input("You: ").strip()
    if user_query.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!\n")
            break
    if not user_query:
            continue
    response = chat(user_query)
    print(f"\nAI: {response}\n")