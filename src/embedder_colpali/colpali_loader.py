import os
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModel
import torch