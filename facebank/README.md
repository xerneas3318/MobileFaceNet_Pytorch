# Facebank – reference faces for recognition

The **facebank** is the set of reference people the camera and video demos recognize. Each person has a folder of one or more face photos; the code builds one embedding per person from those photos.

## 1. Add reference photos

- Create **one subfolder per person** inside `facebank/`.
- Folder name = display name (e.g. `Alice`, `Bob`).
- Put **one or more photos** of that person’s face in the folder (e.g. `.jpg`).

Example:

```
facebank/
  Alice/
    photo1.jpg
    photo2.jpg
  Bob/
    bob_face.jpg
  Howard/
    2019-06-04-12-52-10.jpg
```

Tips:

- Clear, front-facing faces work best.
- You can use `take_picture.py -n Alice` (webcam) or `Take_ID.py -i path/to/photo.jpg -n Alice` to capture or add a photo for a new person.

## 2. Build the embeddings

From the **project root** run the demo **with** the update flag so it rebuilds the facebank from the current photos:

**From camera:**

```bash
python cam_demo.py -u
```

**From video (optional, same facebank):**

```bash
python Video_demo.py --video path/to/video.mp4 -u
```

- `-u` / `--update`: load all images in each `facebank/<Name>/` folder, run them through the model, average embeddings per person, and save:
  - `facebank/facebank.pth` (embeddings)
  - `facebank/names.npy` (names)

After that, run the demos **without** `-u` to use the saved facebank (faster startup).

## 3. Run detection

- **Camera:**  
  `python cam_demo.py`  
  (On Wayland, use: `QT_QPA_PLATFORM=xcb python cam_demo.py`)

- **Video file:**  
  `python Video_demo.py --video /path/to/your/video.mp4`  
  Output is written to `output.avi` by default, or set `--output out.mp4`.

- **Single image:**  
  `python MTCNN_MobileFaceNet.py -img /path/to/image.jpg`

Recognition uses the embeddings in `facebank.pth`; add or change photos in `facebank/<Name>/`, then run again with `-u` to refresh.
