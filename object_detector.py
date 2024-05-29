from ultralytics import YOLO
from flask import request, Flask, send_file
from waitress import serve
from PIL import Image, ImageDraw
from io import BytesIO

from pyngrok import ngrok
from flask_cors import CORS


app = Flask(__name__)
ngrok.set_auth_token("2AdBAXnWv0yZeCSB6jjLkvYYmyV_kUeim1L1oKdrc1p1XZNV")
public_url = ngrok.connect(5000).public_url
CORS(app)

@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint.
    Receives uploaded file with a name "image_file", passes it
    through YOLOv8 object detection network, and returns the image
    with bounding boxes drawn on it.
    :return: The image with bounding boxes as a file response.
    """
    buf = request.files["image_file"]
    img_with_boxes = detect_objects_on_image(buf.stream)
    
    # Save the image to a BytesIO object
    img_io = BytesIO()
    img_with_boxes.save(img_io, format="JPEG")
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network,
    and draws bounding boxes on the image.
    :param buf: Input image file stream
    :return: Image with bounding boxes drawn on it.
    """
    model = YOLO("best.pt")
    image = Image.open(buf)
    results = model.predict(image)
    result = results[0]

    # Draw bounding boxes on image
    draw = ImageDraw.Draw(image)
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        prob_percentage = f"{prob * 100:.2f}%"
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1), f"{result.names[class_id]} {prob_percentage}", fill="green")
    
    return image

print(public_url)
app.run()
