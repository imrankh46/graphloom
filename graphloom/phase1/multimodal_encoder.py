"""
Multimodal Encoder for GraphLoom Phase 1.

This module provides the multimodal understanding component using Qwen3-VL-Instruct
to generate unified factual descriptions from images and questions.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np


class BaseMultimodalEncoder(ABC):
    """Abstract base class for multimodal encoders."""
    
    @abstractmethod
    def encode(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        **kwargs: Any,
    ) -> str:
        """
        Generate a unified factual description from image and question.
        
        Args:
            image: Image path, URL, or numpy array
            question: The question about the image
            **kwargs: Additional arguments
            
        Returns:
            Unified factual description text
        """
        pass


class Qwen3VLEncoder(BaseMultimodalEncoder):
    """
    Multimodal encoder using Qwen3-VL-Instruct.
    
    Generates unified, factual descriptions that consolidate information
    from both the image and the question for downstream triple extraction.
    """
    
    DESCRIPTION_PROMPT = """You are a precise visual analyst. Given an image and a question, 
provide a clear, factual description that:
1. Describes the relevant visual elements in the image
2. Identifies key entities, objects, and their relationships
3. Notes any text, numbers, or labels visible in the image
4. Focuses on information relevant to answering the question

Be concise and factual. Do not speculate or add information not visible in the image.

Question: {question}

Provide your factual description:"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """
        Initialize the Qwen3-VL encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('auto', 'cuda', 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
    
    def _load_model(self) -> None:
        """Lazy load the model and processor."""
        if self._model is not None:
            return
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
            )
        except ImportError as e:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch qwen-vl-utils"
            ) from e
    
    def encode(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        **kwargs: Any,
    ) -> str:
        """
        Generate a unified factual description from image and question.
        
        Args:
            image: Image path, URL, or numpy array
            question: The question about the image
            **kwargs: Additional generation arguments
            
        Returns:
            Unified factual description text
        """
        self._load_model()
        
        prompt = self.DESCRIPTION_PROMPT.format(question=question)
        
        if isinstance(image, (str, Path)):
            image_content = {"type": "image", "image": str(image)}
        else:
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_content = {"type": "image", "image": image}
        
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        
        import torch
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=self.temperature > 0,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text.strip()


class MockMultimodalEncoder(BaseMultimodalEncoder):
    """
    Mock encoder for testing without GPU/model dependencies.
    
    Returns a template description based on the question.
    """
    
    def __init__(self):
        pass
    
    def encode(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        **kwargs: Any,
    ) -> str:
        """Return a mock description for testing."""
        return f"The image shows a scene relevant to the question: {question}. " \
               f"Key visual elements are present that relate to the query."


def create_multimodal_encoder(
    encoder_type: str = "qwen3vl",
    **kwargs: Any,
) -> BaseMultimodalEncoder:
    """
    Factory function to create a multimodal encoder.
    
    Args:
        encoder_type: Type of encoder ('qwen3vl', 'mock')
        **kwargs: Arguments passed to encoder constructor
        
    Returns:
        Configured multimodal encoder instance
    """
    encoders = {
        "qwen3vl": Qwen3VLEncoder,
        "mock": MockMultimodalEncoder,
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available: {list(encoders.keys())}")
    
    return encoders[encoder_type](**kwargs)
