import torch
from comfy.k_diffusion.sampling import default_noise_sampler
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from comfy.model_sampling import EPS
from comfy.samplers import KSAMPLER, calculate_sigmas
from comfy_extras.nodes_model_advanced import ModelSamplingDiscreteDistilled
from tqdm.auto import trange
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview

class ModelSamplingDiscreteDistilledTCD(ModelSamplingDiscreteDistilled, EPS):
    def __init__(self, model_config=None):
        super().__init__(model_config)
        sampling_settings = model_config.sampling_settings if model_config is not None else {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)

        betas = make_beta_schedule(
            beta_schedule, n_timestep=1000, linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0, dtype=torch.float32)
        self.register_buffer("alphas_cumprod", alphas_cumprod.clone().detach())

# 调度器选项
SCHEDULER_NAMES = ["simple", "sgm_uniform"]

class TCDModelSamplingDiscrete:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "scheduler": (SCHEDULER_NAMES, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "eta": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "SAMPLER", "SIGMAS")
    FUNCTION = "patch"

    CATEGORY = "advanced/model"
    
    
    
    
    def patch(self, model, steps=4, scheduler="simple", denoise=1.0, eta=0.3):
        m = model.clone()
        ms = ModelSamplingDiscreteDistilledTCD(model.model.model_config)

        total_steps = steps
        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        elif denoise <= 1.0:
            total_steps = int(steps / denoise)
            sigmas = calculate_sigmas(ms, scheduler, total_steps).cpu()
            sigmas = sigmas[-(steps + 1) :]
        m.add_object_patch("model_sampling", ms)

        timesteps_s = torch.floor((1 - eta) * ms.timestep(sigmas)).to(dtype=torch.long).detach()
        timesteps_s[-1] = 0
        alpha_prod_s = ms.alphas_cumprod[timesteps_s]
        sampler = KSAMPLER(sample_tcd, extra_options={"eta": eta, "alpha_prod_s": alpha_prod_s}, inpaint_options={})
        return (m, sampler, sigmas)

@torch.no_grad()
def sample_tcd(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    eta=0.3,
    alpha_prod_s: torch.Tensor = None,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    beta_prod_s = 1 - alpha_prod_s
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        eps = (x - denoised) / sigmas[i]
        denoised = alpha_prod_s[i + 1].sqrt() * denoised + beta_prod_s[i + 1].sqrt() * eps

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        x = denoised
        if eta > 0 and sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x / alpha_prod_s[i + 1].sqrt() + noise * (sigmas[i + 1] ** 2 + 1 - 1 / alpha_prod_s[i + 1]).sqrt()
        else:
            x = x * (sigmas[i + 1] ** 2 + 1).sqrt()

    return x

class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

class TCD采样器:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL",),
                "正面条件": ("CONDITIONING",),
                "Latent": ("LATENT",),
                "步数": ("INT", {"default": 4, "min": 1, "max": 50}),
                "调度器": (SCHEDULER_NAMES, {"default": "simple"}),
                "降噪强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ETA参数": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "添加噪波": ("BOOLEAN", {"default": True}),
                "噪波种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "CFG强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "负面条件": ("CONDITIONING",),
            }
        }


    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("Latent", "降噪Latent", "信息")
    FUNCTION = "sample"
    CATEGORY = "⛰️CR工具"
    

    def sample(self, 模型, 正面条件, Latent, 步数, 调度器, 降噪强度, ETA参数, 添加噪波, 噪波种子, CFG强度, 负面条件=None):
        # 应用TCD模型采样
        tcd_node = TCDModelSamplingDiscrete()
        m, sampler, sigmas = tcd_node.patch(模型, 步数, 调度器, 降噪强度, ETA参数)

        # 创建CFGGuider
        guider = comfy.samplers.CFGGuider(m)
        if 负面条件 is None:
            # 如果没有提供负面条件，创建一个空的负面条件
            negative = []
        else:
            negative = 负面条件
            
        guider.set_conds(正面条件, negative)
        guider.set_cfg(CFG强度)

        # 准备噪声
        if 添加噪波:
            noise = Noise_RandomNoise(噪波种子)
        else:
            noise = Noise_EmptyNoise()

        # 处理latent图像
        latent = Latent
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        # 采样
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=噪波种子)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        # 生成信息
        info = f"步数={步数}, 调度器={调度器}, 降噪强度={降噪强度}, ETA={ETA参数}, CFG={CFG强度}, 种子={噪波种子}"

        return (out, out_denoised, info)

NODE_CLASS_MAPPINGS = {
    "TCD采样器": TCD采样器,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TCD采样器": "🌀TCD采样器",
}