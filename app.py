import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# --- Configuración de Página ---
st.set_page_config(page_title="RAG con PDF", layout="centered")

# --- Estilos CSS ---
st.markdown("""
<style>
    /* Layout centrado y padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }
    /* Título */
    h1 {
        text-align: center;
        color: #1a1a1a;
    }
    /* Imagen estática */
    .static-image img {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 300px; /* Ajustado para el layout centrado */
    }
    /* Contenedores de secciones (tarjetas) */
    .section-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.04);
        border: 1px solid #e6e6e6;
        margin-bottom: 20px;
    }
    /* Cuadro de respuesta */
    .response-box {
        background-color: #f0f8ff; /* Celeste claro */
        border: 1px solid #cce5ff;
        border-radius: 8px;
        padding: 15px;
        color: #004085; /* Azul oscuro */
    }
    /* Texto de versión de Python */
    [data-testid="stText"] {
        text-align: center;
        font-size: 0.8rem;
        color: #7f8c8d;
        margin-top: 20px;
    }
    /* Subtítulos de sección */
    h3 {
        color: #333;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 5px;
    }
    /* Descripción/Subtítulo principal (movido del sidebar) */
    .stSubheader {
        text-align: center;
        color: #34495e;
        font-style: italic;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Título e Imagen ---
st.title('Generación Aumentada por Recuperación (RAG) . 𖦹˙—')

try:
    image = Image.open('Charuca Cute Robots.jpeg')
    # Aplicar clase CSS
    st.markdown('<div class="static-image">', unsafe_allow_html=True)
    st.image(image, width=350) # El ancho en CSS puede anular esto
    st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen 'Charuca Cute Robots.jpeg': {e}")

# Descripción (movida desde el sidebar)
st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado . ݁₊ ⊹ . ݁˖ . ݁")
st.write("---")

ke = None
with st.container(): # CORREGIDO: "border=False" eliminado
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Paso 1: Ingresa tu Clave de OpenAI")
    ke_input = st.text_input('Ingresa tu Clave de OpenAI', type="password", label_visibility="collapsed", placeholder="sk-...")
    
    if not ke_input:
        st.info("Por favor ingresa tu clave de API de OpenAI para continuar.")
    else:
        os.environ['OPENAI_API_KEY'] = ke_input
        ke = ke_input # Asignamos la clave si es válida
        st.success("¡Clave de API recibida!")
    
    st.markdown('</div>', unsafe_allow_html=True)


# --- PASO 2: Carga de PDF ---
pdf = None
if ke: # Solo mostrar carga si hay clave
    with st.container(): # CORREGIDO: "border=False" eliminado
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("📄 Paso 2: Carga tu archivo PDF")
        pdf = st.file_uploader("Carga el archivo PDF", type="pdf", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PASO 3: Procesamiento y Q&A ---
if pdf is not None and ke:
    with st.container(): # CORREGIDO: "border=False" eliminado
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("❓ Paso 3: Pregunta al Documento")
        
        try:
            # Extraer texto del PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Dividir texto en fragmentos
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            with st.expander("Ver detalles del procesamiento"):
                st.info(f"Texto extraído: {len(text)} caracteres")
                st.success(f"Documento dividido en {len(chunks)} fragmentos")
            
            # Crear embeddings y base de conocimiento
            # Usar st.cache_resource para no recalcular esto cada vez
            @st.cache_resource
            def create_knowledge_base(_chunks):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(_chunks, embeddings)
                return knowledge_base
            
            knowledge_base = create_knowledge_base(chunks)
            
            # Interfaz de pregunta
            user_question = st.text_area("Escribe qué quieres saber sobre el documento", placeholder="Escribe tu pregunta aquí...")
            
            # Procesar pregunta con un botón
            if st.button("Obtener Respuesta", type="primary"):
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    
                    llm = OpenAI(temperature=0, model_name="gpt-4o")
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # Mostrar un spinner mientras se procesa
                    with st.spinner('Pensando...'):
                        response = chain.run(input_documents=docs, question=user_question)
                    
                    # Mostrar respuesta
                    st.markdown("### 💡 Respuesta:")
                    st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
                else:
                    st.warning("Por favor, escribe una pregunta.")
                    
        except Exception as e:
            st.error(f"Error al procesar el PDF: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            
        st.markdown('</div>', unsafe_allow_html=True)
elif ke and pdf is None:
    st.info("Por favor carga un archivo PDF para comenzar")

# --- Versión de Python ---
st.text(f"Versión de Python: {platform.python_version()}")
