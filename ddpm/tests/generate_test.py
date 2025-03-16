from generate import Generate
from config import Config
import matplotlib.pyplot as plt


def test_generate():

    config = Config(
        model_path="./model.pth", # path to the trained model
        num_steps=400,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
        in_channels=1,
        image_shape=28,
        num_time_steps=400
    )

    generator = Generate(config)
    # generate an image
    generated_img = generator.forward(),,,
    generated_img = generated_img.squeeze(0).squeeze(0).numpy()  # remove batch & channel dim

    # visualize the generated image
    plt.figure(figsize=(4, 4))
    plt.imshow(generated_img, cmap="gray")
    plt.axis("off")
    plt.title("Generated Image")
    plt.show()


test_generate()
