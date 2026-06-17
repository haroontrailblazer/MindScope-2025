FROM python:3.13.8-slim
WORKDIR /main
COPY . /main
RUN pip install --no-cache-dir -r requirements.txt

# Inject SEO / Open Graph meta into Streamlit's served index.html so social
# crawlers (which don't execute JS) can read the share-card tags.
RUN python -c "import streamlit, pathlib; idx = pathlib.Path(streamlit.__file__).parent / 'static' / 'index.html'; tags = pathlib.Path('og_meta.html').read_text(encoding='utf-8'); html = idx.read_text(encoding='utf-8'); idx.write_text(html.replace('</head>', tags + '</head>', 1), encoding='utf-8')"

EXPOSE ${PORT}
CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --client.showErrorDetails=false"]