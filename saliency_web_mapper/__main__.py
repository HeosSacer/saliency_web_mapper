from saliency_web_mapper.image_retriever import WebImageRetriever
from saliency_web_mapper.config.environment import SaliencyWebMapperEnvironment

def app(env: SaliencyWebMapperEnvironment):
    imageRetriever = WebImageRetriever()