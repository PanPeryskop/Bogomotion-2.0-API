from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langdetect import detect
from deep_translator import GoogleTranslator

MODEL_PATH = "models/llama-2-7b-chat.Q8_0.gguf"


class BogoLlm:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self) -> LlamaCpp:
        callback = CallbackManager([StreamingStdOutCallbackHandler()])
        n_gpu_layers = 40
        n_batch = 512
        Llama_model: LlamaCpp = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.5,
            max_tokens=124,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            top_p=1,
            callback_manager=callback,
            verbose=True
        )

        return Llama_model

    def generate(self, prompt):
        lang = detect(prompt)
        if lang != 'en':
            prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
        response = self.model.invoke(prompt)
        output = response.replace("Answer: ", "", 1)
        if lang != 'en':
            output = GoogleTranslator(source='en', target='pl').translate(output)

        return output