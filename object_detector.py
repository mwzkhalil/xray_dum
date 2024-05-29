from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image, ImageDraw
from io import BytesIO
import base64

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
    through YOLOv8 object detection network, and returns an array
    of bounding boxes.
    :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes, img_with_boxes = detect_objects_on_image(buf.stream)
    
    # Convert image to base64
    buffered = BytesIO()
    img_with_boxes.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({"boxes": boxes, "image": img_str})

def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network,
    and returns an array of detected objects
    and their bounding boxes.
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")
    image = Image.open(buf)
    results = model.predict(image)
    result = results[0]
    output = []

    # Draw bounding boxes on image
    draw = ImageDraw.Draw(image)
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        prob_percentage = f"{prob * 100:.2f}%"
        output.append([x1, y1, x2, y2, result.names[class_id], prob_percentage])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{result.names[class_id]} {prob_percentage}", fill="red")
    
    return output, image

print(public_url)
app.run()
