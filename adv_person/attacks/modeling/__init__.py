from .camera import *
from .texture import *
from .color_transform import *
from .image_synthesizer import *
from .light import *
from .load_model import *


def update_person_texture(
    person: PersonModel,
    tex_tshirt: CamouflageTexture,
    tex_trouser: CamouflageTexture,
    tau=0.3,
    determinate=False,
):
    person.tshirt.update_texture(tex_tshirt.forward(tau=tau, determinate=determinate))
    person.trouser.update_texture(tex_trouser.forward(tau=tau, determinate=determinate))


class RenderState(Module):
    """Gather everything for render."""

    def __init__(
        self,
        person: PersonModel,
        camera_sampler: CameraSampler,
        light_sampler: LightSampler,
        textures: list[ITexture],
        patch_crops: list[Module],
        img_synthesizer: ImageSynthesizer,
    ) -> None:
        super().__init__()
        self.person = person
        self.camera_sampler = camera_sampler
        self.light_sampler = light_sampler
        self.textures = textures
        self.patch_crops = patch_crops
        self.img_synthesizer = img_synthesizer

    def forward(
        self,
        data: Tensor,
        resample=False,
        is_test=False,
        share_texture=False,
        tex_kwargs={},
        render_kwargs={},
    ):
        if resample:
            self.cameras = self.camera_sampler.sample(len(data))
            self.lights = self.light_sampler.sample()

        # Texture tensors are logged
        self.tex_maps: list[Tensor] = []
        for i, cloth in enumerate(self.person.clothes):
            texture = self.textures[i]
            patch_crop = self.patch_crops[i * 2 + int(is_test)]
            if share_texture:
                if i == 0:
                    tex = texture.forward(**tex_kwargs)
                    shared_tex = tex
                else:
                    tex = shared_tex
            else:
                tex = texture.forward(**tex_kwargs)
            tex: Tensor = patch_crop(tex.squeeze(0)).unsqueeze(0)
            cloth.update_texture(tex)
            self.tex_maps.append(tex)
        rendered = self.person.render(
            self.cameras, self.lights, len(data), **render_kwargs
        )
        return self.img_synthesizer.forward(data, rendered)

    def clamp_(self, clamp_shift=0.0):
        for texture in self.textures:
            texture.clamp_(clamp_shift)
