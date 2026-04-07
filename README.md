# Handwriting-Recognition-Model
Machine learning-based OCR for line-level transcription of historical handwritten texts, focused on Computer Vision and Cultural Heritage using the Washington Database. 

Because the Washington Database uses a specific encoding format (like `s_pt` for periods, `|` for spaces, and `-` for character separation), the very first step is writing a custom parser, followed by a PyTorch CRNN architecture, and finally deploying it.

We have prepared two files:
1. **A Complete Google Colab Guide:** A step-by-step markdown notebook containing the PyTorch code for data parsing, the CRNN model, and the CTC training loop.
2. **A Streamlit Deployment Script:** The complete `app.py` file to serve your trained model as a web application.

### How to use these files:

1. **Training (Colab):** Open Google Colab, upload the notebook to the environment.
2. Ensure your `washingtondb-v1.0.zip` is uploaded, run all the cells, and download the resulting `washington_crnn.pth` and `vocab.pth` files at the end.
3. **Deployment (Streamlit):** * Create a new folder on your computer.
    * Place `app.py`, `washington_crnn.pth`, and `vocab.pth` in that folder.
    * Open your terminal in that folder, run `pip install streamlit torch torchvision opencv-python Pillow`, and then run `streamlit run app.py`. 

This setup provides a professional, end-to-end pipeline covering data engineering, deep learning, and user-facing software deployment!
