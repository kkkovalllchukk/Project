from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from match import KNeighborsVC
import torch
from pathlib import Path
import json
"""
Створює і повертає об’єкт KNeighborsVC для перетворення голосу за допомогою методу k-найближчих сусідів (kNN).
Завантажуються моделі WavLM та HiFiGAN, які використовуються в якості кодера та декодера.
"""
def create_knn_voice_changer(is_pretrained=True, show_progress=True, is_prematched=True, computation_device='cuda') -> KNeighborsVC:
    """
    Використовує вокодер, навчений на `prematched` даних.
    """
    hifigan_model, hifigan_config = load_hifigan_model(is_pretrained, show_progress, is_prematched, computation_device)
    wavlm_model = load_wavlm_model(is_pretrained, show_progress, computation_device)
    voice_changer = KNeighborsVC(wavlm_model, hifigan_model, hifigan_config, computation_device)
    return voice_changer

"""
Завантажує та повертає попередньо навчену модель HiFiGAN для вокодування ознак WavLM.
"""
def load_hifigan_model(is_pretrained=True, show_progress=True, is_prematched=True, computation_device='cuda') -> HiFiGAN:
    """
    is_pretrained - визначає, чи будуть завантажені попередньо навчені ваги моделі HiFiGAN.
    show_progress - визначає, чи буде відображатися прогрес завантаження ваг моделі.
    is_prematched - визначає, чи будуть використовуватися ваги, навчені на даних, які вже були співставлені.
    computation_device - визначає, на якому пристрої будуть виконуватися обчислення.
    """
    config_file_path = Path(__file__).parent.absolute()

    # Load HiFiGAN configuration
    with open(config_file_path/'hifigan'/'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    hifigan_config = AttrDict(json_config)
    computation_device = torch.device(computation_device)

    # Initialize HiFiGAN generator
    generator_model = HiFiGAN(hifigan_config).to(computation_device)
    
    if is_pretrained:
        model_weights_url = "[./prematch_g_02500000.pt]" if is_prematched else "[./g_02500000.pt]"
        state_dict_generator = torch.hub.load_state_dict_from_url(
            model_weights_url,
            map_location=computation_device,
            progress=show_progress
        )
        generator_model.load_state_dict(state_dict_generator['generator'])
    generator_model.eval()
    generator_model.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator_model.parameters()]):,d} parameters.")
    return generator_model
