def sam(boxes: list[list]) -> None:
    """Call SAM (Segment Anything Model) to detect objects.

    Args:
    ----
        boxes (list[list]): A list of arrays of bounding box coordinates.

    """
    pass


def inpainting(prompt: str) -> None:
    """Call stable diffusion inpainting model to edit images.

    Args:
    ----
        prompt (str): An instruction for image editing.

    """
    pass


def grounding_dino(caption: str, box_threshold: float = 0.3, text_threshold: float = 0.25) -> None:
    """Detect and localize objects using text prompt with Grounding DINO.

    Args:
    ----
        caption (str): Text description of objects to detect.
        box_threshold (float): Confidence threshold for bounding boxes.
        text_threshold (float): Confidence threshold for text matching.

    """
    pass


def grounding_dino_sam(caption: str, box_threshold: float = 0.3, text_threshold: float = 0.25) -> None:
    """Detect objects with Grounding DINO and segment with SAM.

    Args:
    ----
        caption (str): Text description of objects to detect and segment.
        box_threshold (float): Confidence threshold for bounding boxes.
        text_threshold (float): Confidence threshold for text matching.

    """
    pass


def instruct_pix2pix(prompt: str) -> None:
    """Edit image using natural language instructions with InstructPix2Pix.

    Args:
    ----
        prompt (str): Instruction describing how to edit the image.

    """
    pass


def ram_grounding_dino() -> None:
    """Recognize anything with RAM and detect with Grounding DINO.

    Uses image directly - no parameters required.

    """
    pass


def seem(refimg: str = None, refmask: str = None) -> None:
    """Segment Everything Everywhere All at Once.

    Args:
    ----
        refimg (str): Reference image path (optional).
        refmask (str): Reference mask in RGBA format (optional).

    """
    pass


def semantic_sam(point: list = None, boxes: list = None) -> None:
    """Perform semantic segmentation with Semantic-SAM.

    Args:
    ----
        point (list): Point coordinates for segmentation (optional).
        boxes (list): Bounding boxes converted to center points (optional).

    """
    pass


def stable_diffusion(prompt: str) -> None:
    """Generate or edit images with Stable Diffusion.

    Args:
    ----
        prompt (str): Text prompt for image generation.

    """
    pass


def blip2_grounding_dino() -> None:
    """Caption with BLIP-2 and detect with Grounding DINO.

    Uses image directly for captioning - no parameters required.

    """
    pass


def clip(text: str = None) -> None:
    """Match image-text pairs using CLIP.

    Args:
    ----
        text (str): Text to match against the image (optional).

    """
    pass


def openseed(mode: str = None, prompt: dict = None) -> None:
    """Perform segmentation with OpenSEED.

    Args:
    ----
        mode (str): Either 'openseed' or 'controlnet' (optional).
        prompt (dict): Dictionary containing caption field (optional).

    """
    pass


def controlnet(prompt: str = None, mask: str = None) -> None:
    """Generate images with ControlNet using mask conditioning.

    Args:
    ----
        prompt (str): Text prompt or caption for image generation (optional).
        mask (str): Mask image for conditioning (optional).

    """
    pass
