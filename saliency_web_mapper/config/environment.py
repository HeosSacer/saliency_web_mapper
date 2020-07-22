from saliency_web_mapper.config.typesafe_dataclass import TypesafeDataclass
from typing import List, Dict, Tuple, Sequence


class SaliencyWebMapperEnvironment(TypesafeDataclass):
    # Defaults with type


    # Debug
    debug: bool = False
    debug_address: str = "192.168.178.33"

    def __init__(self):
        super().__init__()
