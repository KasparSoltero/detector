import torch
import torchaudio
import numpy as np
from PIL import Image
from ultralytics import YOLO
from matplotlib.colors import hsv_to_rgb
import random

def load_spectrogram_chunks(path, chunk_length=10, overlap=0.5, resample_rate=48000):
    waveform, sample_rate = torchaudio.load(path)
    resample = torchaudio.transforms.Resample(sample_rate, resample_rate)
    waveform = resample(waveform)

    # Calculate the number of samples per chunk
    samples_per_chunk = int(chunk_length * resample_rate)

    # Calculate the number of samples to overlap
    overlap_samples = int(samples_per_chunk * overlap)

    # Split the waveform into n 10-second chunks with 50% overlap
    length = waveform.shape[1]
    n = int(2*length / (samples_per_chunk)) - 1 # only works for 0.5 overlap
    if n>20: n=20
    chunks = []
    start = 0
    end = samples_per_chunk
    for _ in range(n):
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        start += samples_per_chunk - overlap_samples
        end += samples_per_chunk - overlap_samples

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0
    )
    return [spec_transform(chunk) for chunk in chunks]

def spec_to_image(spectrogram, image_normalise=0.2):
    # make ipeg image
    spec = np.squeeze(spectrogram.numpy())
    # spec = (spec - np.min(spec)) / (np.ptp(spec)) #normalise to 0-1 # the max is typically an outlier so this normalisation is destructive
    spec = spec * image_normalise / spec.mean() # normalise to 0.2 mean
    spec = np.flipud(spec) #vertical flipping for image cos

    image_width, image_height = spec.shape[1], spec.shape[0]
    image = Image.fromarray(np.uint8(spec * 255), 'L')
    image = image.resize((640, 640), Image.Resampling.LANCZOS) #resize for yolo
    return image

def get_detections(paths, model_no):
    model = YOLO(f'/Users/kaspar/Documents/spectral_detector/runs/detect/train{model_no}/weights/best.pt')
    all_boxes = []
    for path in paths:
        spectrograms = load_spectrogram_chunks(path)
        images = [spec_to_image(spec) for spec in spectrograms]
        results = model.predict(images, save=False, show=False, device='mps')
        boxes = []
        for i, result in enumerate(results):
            for box_set in result.boxes: 
                for box in box_set:
                    x1, y1, x2, y2 = np.squeeze(box.xyxyn.to('cpu'))
                    x1 = ((5*i)+(float(x1)*10)) / 55
                    x2 = ((5*i)+(float(x2)*10)) / 55
                    y1 = 1-y1
                    y2 = 1-y2
                    boxes.append([x1, y1, x2, y2])
        all_boxes.append(boxes)
    return all_boxes

def pcen(spec, s=0.025, alpha=0.6, delta=0, r=0.2, eps=1e-6):
    """
    Apply Per-Channel Energy Normalization (PCEN) to a spectrogram.
    
    Parameters:
    - spec (Tensor): Input spectrogram.
    - s (float): Time constant.
    - alpha (float): Exponent for the smooth function.
    - delta (float): Bias term.
    - r (float): Root compression parameter.
    - eps (float): Small value to prevent division by zero.
    
    Returns:
    - Tensor: PCEN-processed spectrogram.
    """
    # Handle 4D tensor by flattening out the batch and channel dimensions
    orig_shape = spec.shape # Save original shape
    flattened_spec = spec.view(orig_shape[0] * orig_shape[1], 1, -1) # Flatten N and C into single dimension

    # Apply avg_pool1d
    M = torch.nn.functional.avg_pool1d(flattened_spec, 
                                        kernel_size=int(s * spec.size(-1)), 
                                        stride=1, 
                                        padding=int((s * spec.size(-1) - 1) // 2)).view(orig_shape)
    pcen_spec = (spec / (M + eps).pow(alpha) + delta).pow(r) - delta**r
    return pcen_spec

def spec_to_pil(spec, resize=None, iscomplex=False, normalise='power_to_PCEN', color_mode='HSV'):
    if iscomplex:
        spec = torch.abs(torch.view_as_real(spec)).sum(-1)

    if normalise:
        if normalise == 'power_to_dB':
            spec = 10 * torch.log10(spec + 1e-6)
        elif normalise == 'dB_to_power':
            spec = 10 ** (spec / 10)
        elif normalise == 'power_to_PCEN':
            spec = pcen(spec)
        elif normalise == 'complex_to_PCEN':
            # square complex energy for magnitude power
            spec = torch.square(torch.abs(spec))
            spec = pcen(spec)

    spec = np.squeeze(spec.numpy())
    spec = np.flipud(spec)  # vertical flipping for image cos
    spec = (spec - spec.min()) / (spec.max() - spec.min())  # scale to 0 - 1

    if color_mode == 'HSV':
        value = spec
        saturation = 4 * value * (1 - value)
        hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] # linearly spaced hues over frequency range
        hue = np.tile(hue, (1, spec.shape[1]))
        hsv_spec = np.stack([hue, saturation, value], axis=-1)
        rgb_spec = hsv_to_rgb(hsv_spec)
        spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
    else:
        spec = Image.fromarray(np.uint8(spec * 255), 'L')

    if resize:
        spec = spec.resize(resize, Image.Resampling.LANCZOS)
    
    return spec

def complex_spectrogram_transformed(
        spec,
        lowpass=-1,
        highpass=0,
        scale_intensity=None,
        clamp_intensity=None,
        set_snr_db=None,
        scale_mean=None,
        set_rms=None,
        normalise=None,
        to_torch=False,
        to_numpy=False,
        to_pil=False,
        resize=None
    ):
    # scaling and clamping is done BEFORE normalisation

    real_part = spec.real
    imaginary_part = spec.imag

    if highpass:
        real_part[:, :highpass, :] = 0
        imaginary_part[:, :highpass, :] = 0
    if lowpass:
        real_part[:, lowpass:, :] = 0
        imaginary_part[:, lowpass:, :] = 0

    if isinstance(scale_intensity, int) or isinstance(scale_intensity, float) or (isinstance(scale_intensity, torch.Tensor) and scale_intensity.shape==[]): 
        real_part = real_part * scale_intensity
        imaginary_part = imaginary_part * scale_intensity
    elif isinstance(scale_intensity, tuple) or isinstance(scale_intensity, list):
        minimum = scale_intensity[0]
        maximum = scale_intensity[1]
        real_part = (real_part - real_part.min()) / (real_part.max() - real_part.min()) * (maximum - minimum) + minimum
        imaginary_part = (imaginary_part - imaginary_part.min()) / (imaginary_part.max() - imaginary_part.min()) * (maximum - minimum) + minimum
    if isinstance(clamp_intensity, int) or isinstance(clamp_intensity, float):
        real_part = real_part.clamp(min=clamp_intensity)
        imaginary_part = imaginary_part.clamp(min=clamp_intensity)
    elif isinstance(clamp_intensity, tuple) or isinstance(clamp_intensity, list):
        real_part = real_part.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
        imaginary_part = imaginary_part.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
    if set_snr_db:
        power = torch.mean(torch.sqrt(real_part**2 + imaginary_part**2))
        # set_snr to db
        set_snr = 10 ** (set_snr_db / 10)
        normalise_power = set_snr / power
        real_part = real_part * normalise_power
        imaginary_part = imaginary_part * normalise_power
    
    if scale_mean:
        # normalise mean value for the non-zero values
        non_zero_real = real_part[real_part != 0]
        mean_power = non_zero_real.mean()
        normalise_power = scale_mean / mean_power
        real_part = real_part * normalise_power
        imaginary_part = imaginary_part * normalise_power
    if set_rms:
        component_magnitude = torch.sqrt(real_part**2 + imaginary_part**2)
        rms = torch.sqrt(torch.mean(component_magnitude**2))
        normalise_rms_factor = set_rms / rms
        real_part = real_part * normalise_rms_factor
        imaginary_part = imaginary_part * normalise_rms_factor

    # if normalise:
    #     if normalise=='power_to_dB':
    #         real_part = 10 * torch.log10(real_part + 1e-6)
    #         imaginary_part = 10 * torch.log10(imaginary_part + 1e-6)
    #     elif normalise=='dB_to_power':
    #         real_part = 10 ** (real_part / 10)
    #         imaginary_part = 10 ** (imaginary_part / 10)
    #     elif normalise=='PCEN':
    #         real_part = pcen(real_part)
    #         imaginary_part = pcen(imaginary_part)

    if to_pil:
        return spec_to_pil(real_part + 1j * imaginary_part, resize=(640,640), iscomplex=True, normalise='complex_to_PCEN', color_mode='HSV')

    if to_numpy:
        return np.stack([real_part.numpy(), imaginary_part.numpy()], axis=-1)
    return real_part + 1j * imaginary_part

def spectrogram_transformed(
        spec,
        lowpass=-1, 
        highpass=0, 
        scale_intensity=None, 
        clamp_intensity=None, 
        set_snr_db=None,
        scale_mean=None, 
        set_rms=None,
        normalise='power_to_PCEN', 
        to_torch=False, 
        to_numpy=False, 
        to_pil=False, 
        resize=None
    ):
    # [spec.shape] = [batch, freq_bins, time_bins]
    # scaling and clamping is done BEFORE normalisation
    is_numpy = isinstance(spec, np.ndarray)
    if is_numpy:
        to_numpy=True
        spec = torch.from_numpy(spec)
    if spec.is_complex():
        spec = complex_spectrogram_transformed(
            spec,lowpass=lowpass,highpass=highpass,scale_intensity=scale_intensity,set_snr_db=set_snr_db,clamp_intensity=clamp_intensity,scale_mean=scale_mean,set_rms=set_rms,normalise=normalise,to_torch=to_torch,to_numpy=to_numpy,to_pil=to_pil,resize=resize
        )
    else:
        print(f'real spectrogram')
        if len(spec.shape) == 2:
            spec = spec.unsqueeze(0)

        if highpass:
            spec[:, :highpass, :] = 0
        if lowpass:
            spec[:, lowpass:, :] = 0

        if isinstance(scale_intensity, int) or isinstance(scale_intensity, float) or (isinstance(scale_intensity, torch.Tensor) and scale_intensity.shape==[]):
            spec = spec * scale_intensity
        elif isinstance(scale_intensity, tuple) or isinstance(scale_intensity, list):
            minimum = scale_intensity[0]
            maximum = scale_intensity[1]
            spec = (spec - spec.min()) / (spec.max() - spec.min()) * (maximum - minimum) + minimum
        if isinstance(clamp_intensity, int) or isinstance(clamp_intensity, float):
            spec = spec.clamp(min=clamp_intensity)
        elif isinstance(clamp_intensity, tuple) or isinstance(clamp_intensity, list):
            spec = spec.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
        if scale_mean:
            # normalise mean value for the non-zero values
            non_zero_spec = spec[spec != 0]
            mean_power = non_zero_spec.mean()
            normalise_power = scale_mean / mean_power
            spec = spec * normalise_power
        if set_rms:
            rms = torch.sqrt(torch.mean(torch.square(spec)))
            normalise_rms_factor = set_rms / rms
            spec = spec * normalise_rms_factor

        if to_pil:
            # spec = np.squeeze(spec.numpy())
            # spec = np.flipud(spec) # vertical flipping for image cos
            # spec = (spec - spec.min()) / (spec.max() - spec.min()) # scale to 0 - 1
            # if to_pil=='HSV':
            #     value = spec
            #     saturation = 4 * value * (1 - value)
            #     hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] # linearly spaced hues over frequency range
            #     hue = np.tile(hue, (1, spec.shape[1]))
            #     hsv_spec = np.stack([hue, saturation, value], axis=-1)
            #     rgb_spec = hsv_to_rgb(hsv_spec)
            #     spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
            # else: # greyscale
            #     spec = Image.fromarray(np.uint8(spec * 255), 'L')
            # if resize:
            #     spec = spec.resize(resize, Image.Resampling.LANCZOS)
            # return spec
            return spec_to_pil(spec, resize=resize, iscomplex=False, normalise=normalise,color_mode='HSV')
    
    if to_numpy:
        spec = np.squeeze(spec.numpy())
    if scale_intensity:
        return spec, scale_intensity
    if set_snr_db:
        return spec, set_snr_db
    return spec

def load_spectrogram(
        paths, 
        unit_type='complex', 
        rms_normalised=1, 
        random_crop=None, 
        chunk_length=None, 
        max=20,
        overlap=0.5, 
        resample_rate=48000
    ):
    
    if unit_type == 'complex':
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            power=None,  # Produces complex-valued spectrogram without reducing to power or magnitude
    )
    elif unit_type=='power':
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048, 
            win_length=2048, 
            hop_length=512, 
            power=2.0,
            window_fn=torch.hamming_window
        )
    elif unit_type=='dB':
        spec_transform = lambda x: 10 * torch.log10(torch.abs(torchaudio.transforms.Spectrogram(
            n_fft=2048, 
            win_length=2048, 
            hop_length=512, 
            power=2.0,
            window_fn=torch.hamming_window
        )(x)) + 1e-6)

    specs = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        waveform, sample_rate = torchaudio.load(path)
        resample = torchaudio.transforms.Resample(sample_rate, resample_rate)
        waveform = resample(waveform)
        if waveform.shape[0]>1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # mono

        #randomly crop to random_crop seconds
        if random_crop: 
            cropped_samples = random_crop * resample_rate
            if waveform.shape[1] > cropped_samples:
                start = random.randint(0, waveform.shape[1] - cropped_samples)
                waveform = waveform[:, start:start+cropped_samples]
            else:
                print(f"Error: {path} is shorter than random crop length {random_crop}")
                return None
        
        # rms normalise
        if rms_normalised:
            rms = torch.sqrt(torch.mean(torch.square(waveform)))
            waveform = waveform*rms_normalised/rms

        # separate into chunks of chunk_length seconds
        if chunk_length and (waveform.shape[1] > (chunk_length * resample_rate)):
            samples_per_chunk = int(chunk_length * resample_rate)
            overlap_samples = int(samples_per_chunk * overlap)
            samples_overlap_difference = samples_per_chunk - overlap_samples
            for i in range(0, waveform.shape[1], samples_overlap_difference):
                chunk = waveform[:, i:i+samples_per_chunk]
                spec = spec_transform(chunk)
                specs.append(spec)
                if len(specs)>=max:
                    break

        else:
            spec = spec_transform(waveform)
            specs.append(spec)
    
    if len(specs)==1:
        return specs[0]
    return specs
