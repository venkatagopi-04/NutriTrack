from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from sqlalchemy import text
from werkzeug.utils import secure_filename
from PIL import Image
import google.generativeai as genai
import json

# Load API key from config.json




app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@127.0.0.1/nutri'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
UPLOAD_FOLDER = 'static/recent_images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

modell = tf.keras.models.load_model('./foodnutri.h5')


db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class UserNutrition(db.Model):
    username = db.Column(db.String(30), primary_key=True, nullable=False)
    month_target = db.Column(db.Integer)
    c_month_target = db.Column(db.Integer)
    week_target = db.Column(db.Integer)
    c_week_target = db.Column(db.Integer)
    carbs = db.Column(db.Integer)
    proteins = db.Column(db.Integer)
    fats = db.Column(db.Integer)
    sugars = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())




with open("config.json", "r") as config_file:
    config = json.load(config_file)

api_key = config.get("GENAI_API_KEY")

# Configure API key
genai.configure(api_key=api_key)

# Define the model
model = genai.GenerativeModel("gemini-1.5-flash")



# Routes
@app.route('/')
def index():
    return render_template('index.html')  # Render the login/signup page

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('login-email')  # Use email for login
    password = request.form.get('login-password')

    # Fetch user by email
    user = User.query.filter_by(email=email).first()

    if user and user.password == password:  # Direct password comparison
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard', email=email))

    else:
        flash('Invalid email or password', 'danger')
        return redirect(url_for('index'))

@app.route('/signup', methods=['POST'])
def signup():
    fullname = request.form.get('signup-fullname')
    mobile = request.form.get('signup-mobile')
    email = request.form.get('signup-email')
    password = request.form.get('signup-password')
    confirm_password = request.form.get('signup-confirm-password')

    # Check if passwords match
    if password != confirm_password:
        flash('Passwords do not match', 'danger')
        return redirect(url_for('index'))

    # Check if user already exists
    user_exists = User.query.filter((User.email == email) | (User.mobile == mobile)).first()
    if user_exists:
        flash('User with this email or mobile number already exists', 'danger')
        return redirect(url_for('index'))

    # Insert user into the database (no hashing of password)
    new_user = User(fullname=fullname, mobile=mobile, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()

    flash('Signup successful! Please login.', 'success')
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    email = request.args.get('email')

    if not email:
        flash("Invalid email.", "danger")
        return redirect(url_for('login'))

    today = datetime.now()
    last_week_date = today - timedelta(days=7)
    last_month_date = today - timedelta(days=30)

    # Fetch weekly data using raw SQL
    week_query = text("""
        SELECT * FROM user_nutrition
        WHERE username = :email AND timestamp >= :last_week
    """)
    week_data = db.session.execute(week_query, {"email": email, "last_week": last_week_date}).fetchall()

    # Fetch monthly data using raw SQL
    month_query = text("""
        SELECT * FROM user_nutrition
        WHERE username = :email AND timestamp >= :last_month
    """)
    month_data = db.session.execute(month_query, {"email": email, "last_month": last_month_date}).fetchall()

    # Fetch all user data to get the latest record
    all_query = text("""
        SELECT * FROM user_nutrition WHERE username = :email ORDER BY timestamp DESC
    """)
    all_data = db.session.execute(all_query, {"email": email}).fetchall()

    if all_data:
        latest_record = all_data[0]  # Last entry (most recent)
        
        # Calculate weekly sums
        carbs_sum_week = sum(entry.carbs for entry in week_data)
        proteins_sum_week = sum(entry.proteins for entry in week_data)
        fats_sum_week = sum(entry.fats for entry in week_data)
        sugars_sum_week = sum(entry.sugars for entry in week_data)

        current_week_value = carbs_sum_week + proteins_sum_week + fats_sum_week + sugars_sum_week
        target_week_value = latest_record.month_target / 4  # Assuming monthly target is divided by 4 weeks

        # Calculate monthly sums
        carbs_sum_month = sum(entry.carbs for entry in month_data)
        proteins_sum_month = sum(entry.proteins for entry in month_data)
        fats_sum_month = sum(entry.fats for entry in month_data)
        sugars_sum_month = sum(entry.sugars for entry in month_data)

        current_month_value = carbs_sum_month + proteins_sum_month + fats_sum_month + sugars_sum_month
        target_month_value = latest_record.month_target

        # Monthly individual values
        c_values = {
            "C1": latest_record.carbs,
            "C2": latest_record.proteins,
            "C3": latest_record.fats,
            "C4": latest_record.sugars
        }

        return render_template(
            'dashboard.html',
            user_data=latest_record,
            current_week_value=current_week_value,
            target_week_value=target_week_value,
            c_values=c_values,
            carbs_sum_week=carbs_sum_week,
            proteins_sum_week=proteins_sum_week,
            fats_sum_week=fats_sum_week,
            sugars_sum_week=sugars_sum_week,
            current_month_value=current_month_value,
            target_month_value=target_month_value,
            carbs_sum_month=carbs_sum_month,
            proteins_sum_month=proteins_sum_month,
            fats_sum_month=fats_sum_month,
            sugars_sum_month=sugars_sum_month
        )
    else:
        flash("User data not found.", "danger")
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))  # No session to pop, just redirect












@app.route('/submit-targets', methods=['POST'])
def submit_targets():
    username = request.form.get('email')  # Changed 'email' to 'username' for consistency
    month_target = request.form.get('monthlyTarget')

    if not username or not month_target:
        flash("All target fields are required.", "danger")
        return redirect(url_for('dashboard', email=username))

    try:
        month_target = int(month_target)
        week_target = month_target // 4  # Integer division for better accuracy

        if month_target <= 0:
            flash("Targets must be positive numbers.", "danger")
            return redirect(url_for('dashboard', email=username))

        # Initialize other values
        c_month_targets = 0
        c_week_targets = 0
        carbs = 0
        fats = 0
        proteins = 0
        sugars = 0

        # Create new entry for UserNutrition table
        new_target = UserNutrition(
            username=username,
            month_target=month_target,
            c_month_target=c_month_targets,
            week_target=week_target,
            c_week_target=c_week_targets,
            carbs=carbs,
            fats=fats,
            proteins=proteins,
            sugars=sugars
        )

        db.session.add(new_target)
        db.session.commit()

        flash("Targets submitted successfully!", "success")

    except ValueError:
        flash("Invalid input. Please enter numeric values.", "danger")

    return redirect(url_for('dashboard', email=username))


















# Nutrient data per 100g
NUTRIENT_DATA = {
    'pizza': {'carbs': 140, 'proteins': 40, 'fats': 108, 'sugars': 16, 'calories': 280},
    'samosa': {'carbs': 120, 'proteins': 24, 'fats': 135, 'sugars': 8, 'calories': 250},
    'ice_cream': {'carbs': 100, 'proteins': 12, 'fats': 90, 'sugars': 48, 'calories': 210},
    'fried_rice': {'carbs': 160, 'proteins': 24, 'fats': 72, 'sugars': 12, 'calories': 250},
    'chicken_curry': {'carbs': 28, 'proteins': 80, 'fats': 144, 'sugars': 16, 'calories': 220}
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    # Fetch data from the form
    quantity = request.form.get('quantity')
    username = request.form.get('email')
    month_target = request.form.get('m_targets')
    week_target = request.form.get('w_targets')
    c_month_value = 0
    c_week_value = 0

    # Validate quantity
    if not quantity or not quantity.isdigit():
        return jsonify({'error': 'Invalid quantity'}), 400
    quantity = float(quantity)

    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Secure the filename and save the file to the folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open the image for prediction
        image = Image.open(file_path)
        image = image.resize((200, 200))  # Resize to fit your model input
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        

        # Prediction
        prediction = modell.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        class_labels = ['chicken_curry', 'fried_rice', 'ice_cream', 'pizza', 'samosa']
        predicted_label = class_labels[predicted_class[0]]

        if predicted_label in NUTRIENT_DATA:
            nutrients_per_100g = NUTRIENT_DATA[predicted_label]
            scale_factor = quantity / 100
            calculated_nutrients = {key: int(value * scale_factor) for key, value in nutrients_per_100g.items()}

            # Add user nutrition data to the database
            new_entry = UserNutrition(
                username=username,
                month_target=month_target,
                c_month_target=c_month_value,
                week_target=week_target,
                c_week_target=c_week_value,
                carbs=calculated_nutrients['carbs'],
                proteins=calculated_nutrients['proteins'],
                fats=calculated_nutrients['fats'],
                sugars=calculated_nutrients['sugars'],
                timestamp=datetime.now()
            )
            db.session.add(new_entry)
            db.session.commit()

        return redirect(url_for('dashboard', email=username))


@app.route('/recent-images')
def get_recent_images():
    # Get all the image files from the recent_images folder
    images = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    
    # Sort images by creation time and pick the top 3 recent images
    images.sort(key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
    
    # Get the top 3 images
    recent_images = images[:5]
    
    # Return the URLs of the top 3 images
    image_urls = [f'/static/recent_images/{image}' for image in recent_images]
    return jsonify({'recent_images': image_urls}), 200






@app.route("/process_chat", methods=["POST"])
def process_chat():
    try:
        # Get user input
        user_message = request.form.get("user_message", "").strip()
        print(user_message)

        # Get hidden nutritional report
        nutrition_report = request.form.get("nutrition_report", "").strip()
        print(nutrition_report)

        # Merge both into a single prompt
        combined_prompt = f"""
            User: question: {user_message}

            User report: {nutrition_report}

            Instructions:
            - Answer only if the question is about health, nutrition, body, or the provided user report.
            - If the question is not about those topics, respond with: "Ask me only about nutrition and health."
            """
        # Send the combined input to Gemini AI
        response = model.generate_content(combined_prompt)

        # Extract AI-generated text
        ai_response = response.text.strip() if response and hasattr(response, "text") else "No valid response received."

        return jsonify({"response": ai_response})  # Send JSON response back to frontend

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error message

   






















# Initialize the database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)



