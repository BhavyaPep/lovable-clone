# streamlit_app.py
import os
from pathlib import Path
from io import BytesIO
import zipfile
import streamlit as st
from dotenv import load_dotenv

# Import your compiled agent and project tools
# (Assumes streamlit_app.py sits next to graph.py; adjust import path if different)
from graph import agent
from tools import init_project_root, set_project_root

load_dotenv()

st.set_page_config(page_title="Project Generator", page_icon="🛠️", layout="wide")
st.title("🛠️ Project Generator (LangGraph + Azure OpenAI)")

# --- Sidebar controls ---
with st.sidebar:
    recursion_limit = st.number_input(
        "Recursion limit",
        min_value=10, max_value=2000, value=100, step=10,
        help="Max steps for the LangGraph agent before it stops."
    )
    run_btn = st.button("🚀 Generate Project", type="primary", use_container_width=True)

# --- Main input ---
user_prompt = st.text_area(
    "Describe the project you want:",
    height=160,
    placeholder="e.g., Build a FastAPI service with SQLite for a simple order management system…"
)

# Session buckets
if "last_zip" not in st.session_state:
    st.session_state.last_zip = None
    st.session_state.last_zip_name = None
    st.session_state.project_root = None
    st.session_state.final_state = None

def zip_directory(dir_path: Path) -> bytes:
    """Zip a folder into memory and return bytes."""
    mem = BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in dir_path.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(dir_path))
    mem.seek(0)
    return mem.getvalue()

if run_btn:
    if not user_prompt.strip():
        st.error("Please enter a prompt before generating.")
        st.stop()

    with st.spinner("Initializing project and running the agent…"):
        try:
            # 1) Create a fresh project root and set it for the tools
            project_root = init_project_root.invoke({})
            set_project_root(Path(project_root))

            # 2) Run the LangGraph agent
            final_state = agent.invoke(
                {"user_prompt": user_prompt},
                config={"recursion_limit": int(recursion_limit)}
            )

            # 3) Zip the generated folder
            zip_bytes = zip_directory(Path(project_root))

            # 4) Persist in session (survives reruns)
            st.session_state.last_zip = zip_bytes
            st.session_state.last_zip_name = f"{Path(project_root).name}.zip"
            st.session_state.project_root = project_root
            st.session_state.final_state = final_state

        except Exception as e:
            # Friendly surfacing for common Azure content-filter blocks
            msg = str(e)
            if "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg:
                st.warning(
                    "Azure content filter blocked this run. Try rephrasing your prompt "
                    "(avoid words like ‘jailbreak’, ‘bypass’, ‘ignore previous instructions’, etc.)."
                )
            st.exception(e)

# --- Output / download ---
if st.session_state.last_zip:
    st.success(f"✅ Generated at: {st.session_state.project_root}")

    st.download_button(
        "⬇️ Download project as ZIP",
        data=st.session_state.last_zip,
        file_name=st.session_state.last_zip_name,
        mime="application/zip",
        use_container_width=True,
    )

    with st.expander("🔎 Final state (for debugging)"):
        st.json(st.session_state.final_state)

    with st.expander("📁 Generated files"):
        root = Path(st.session_state.project_root)
        files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
        if files:
            st.code("\n".join(files), language="text")
        else:
            st.write("No files found.")