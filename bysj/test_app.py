import streamlit as st
import os

st.title('测试页面')
st.write('如果你能看到这句话，说明Streamlit本身没问题。')

@st.cache_resource
def load_clip():
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # 手动加载本地权重
    state_dict = torch.load(os.path.expanduser('~/.cache/open_clip/ViT-B-32-openai.pt'), map_location='cpu')
    model.load_state_dict(state_dict)
    return model, preprocess, tokenizer 