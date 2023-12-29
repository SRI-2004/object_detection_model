from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class FrameData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frame_number = db.Column(db.Integer)
    class_label = db.Column(db.Float, nullable=False, default=0.0)
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    x_midpoint = db.Column(db.Float, nullable=False, default=0.0)
    y_midpoint = db.Column(db.Float, nullable=False, default=0.0)
    width = db.Column(db.Float, nullable=False, default=0.0)
    height = db.Column(db.Float, nullable=False, default=0.0)

@app.route('/')
def index():
    return jsonify({'message': 'Server is running'})

@app.route('/update', methods=['POST'])
def update_data():
    data = request.get_json()

    for frame_data in data.get('frames', []):
        if 'data' in frame_data and frame_data['data']:
            frame_data_info = frame_data['data'][0]
            new_frame_data = FrameData(
                frame_number=frame_data.get('frame_no', 0),
                class_label=frame_data_info.get('class_label', 0.0),
                confidence=frame_data_info.get('confidence', 0.0),
                x_midpoint=frame_data_info.get('x_midpoint', 0.0),
                y_midpoint=frame_data_info.get('y_midpoint', 0.0),
                width=frame_data_info.get('width', 0.0),
                height=frame_data_info.get('height', 0.0),
            )
            db.session.add(new_frame_data)

    db.session.commit()

    return jsonify({'message': 'Data received successfully!'})

@app.route('/get_data', methods=['GET'])
def get_data():
    data = FrameData.query.all()
    data_list = [{
        'frameNumber': d.frame_number,
        'classLabel': d.class_label,
        'confidence': d.confidence,
        'xMidpoint': d.x_midpoint,
        'yMidpoint': d.y_midpoint,
        'width': d.width,
        'height': d.height,
    } for d in data]

    return jsonify(data_list)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()
