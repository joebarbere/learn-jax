"""A minimal MCP server (Model Context Protocol, Anthropic 2024) exposing Module 38's paper search as a tool.

Any MCP client — Claude Code, Claude Desktop, LM Studio, or the agent in this lesson — can launch it over stdio
and call `search_papers`. It runs on the CPU (JAX_PLATFORMS=cpu is fine: BM25 is one sparse matmul).

    containers/run.sh python modules/39-agents/mcp_server.py      # speaks MCP on stdin/stdout

The owner's aws-ai `local-kb` ships the production-scale version of this idea (search_knowledge_base and
ask_knowledge_base over OpenSearch or pgvector).
"""
import importlib.util
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from learnjax.course import load_exercises
from learnjax.data import radio_abstracts

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("m39_agents_mcp", HERE / "agents.py")
rag = _load("m38_rag_mcp", HERE.parent / "38-rag" / "rag.py")
ex38 = load_exercises(HERE.parent / "38-rag", solutions=True)
papers = A.paper_subset(radio_abstracts())
index = rag.build_index([p["title"] + ". " + p["abstract"] for p in papers], ex38)
search = A.paper_tool(index, papers).fn

server = MCPServer("learn-jax-papers", instructions="Search 4,000 radio-astronomy arXiv papers by title words.")


@server.tool()
def search_papers(query: str) -> str:
    """Search arXiv radio-astronomy papers by title words; returns the top 3 titles with publication dates."""
    return search(query)


if __name__ == "__main__":
    server.run("stdio")
