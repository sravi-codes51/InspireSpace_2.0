from flask import Flask, render_template, request, redirect, url_for, flash, session
import os, json, datetime
from werkzeug.utils import secure_filename
from transformers import pipeline
import cv2

app = Flask(__name__)
app.secret_key = "inspirespace-secret"

UPLOAD_FOLDER = "static/uploads"
DATA_FILE = "database.json"
USERS_FILE = "users.json"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Data functions ---
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

@app.context_processor
def inject_metadata():
    return dict(load_data=load_data, session=session)

# --- Hugging Face AI Models ---
nsfw_detector = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
violence_detector = pipeline("image-classification", model="microsoft/resnet-50")

# --- Violence text filter ---
VIOLENCE_KEYWORDS = ["murder", "blood", "kill", "gore", "fight", "dead", "weapon", "shoot", "stab", "attack"]

def contains_violence_text(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in VIOLENCE_KEYWORDS)

# ---------- ROUTES ---------- #
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/index")
def index():
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template("index.html", files=files)

@app.route("/profile")
def profile():
    if "username" in session:
        return render_template("profile.html", username=session["username"])
    else:
        flash("⚠ Please login first!", "error")
        return redirect(url_for("login"))

# -------- Signup --------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # load existing users
        users = {}
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                users = json.load(f)

        if username in users:
            flash("❌ Username already exists!", "error")
            return redirect(url_for("signup"))

        # save new user
        users[username] = {"email": email, "password": password}
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

        flash(f"✅ Account created for {username}! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# -------- Login --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                users = json.load(f)
        else:
            users = {}

        if username in users and users[username]["password"] == password:
            session["username"] = username
            flash(f"✅ Welcome {username}, you are logged in!", "success")
            return redirect(url_for("index"))
        else:
            flash("❌ Invalid username or password!", "error")

    return render_template("login.html")

# -------- Logout --------
@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("✅ Logged out successfully!", "success")
    return redirect(url_for("welcome"))

# -------- Upload --------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        flash("⚠ You must login to upload!", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["file"]
        creator = request.form["creator"]
        description = request.form["description"]

        if file.filename == "":
            flash("❌ No file selected!", "error")
            return redirect(request.url)

        # ✅ Violence text check (filename + description)
        if contains_violence_text(file.filename) or contains_violence_text(description):
            flash("❌ Upload blocked: Violence-related words in filename/description!", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # --- AI Detection ---
        nsfw_flag, violence_flag = False, False

        # --- If video file → check frame by frame ---
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            cap = cv2.VideoCapture(filepath)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Sample every 30th frame (~1 frame/sec if 30fps)
                if frame_count % 30 == 0:
                    frame_path = f"{filepath}_frame.jpg"
                    cv2.imwrite(frame_path, frame)

                    # Run detectors
                    nsfw_result = nsfw_detector(frame_path)
                    violence_result = violence_detector(frame_path)

                    # Check NSFW
                    if nsfw_result and nsfw_result[0]["label"].lower() == "nsfw":
                        nsfw_flag = True

                    # Check Violence
                    bad_labels = ["assault", "weapon", "blood", "gore", "fight"]
                    if violence_result:
                        for res in violence_result:
                            if res["label"].lower() in bad_labels and res["score"] > 0.6:
                                violence_flag = True

                    os.remove(frame_path)

                    # If flagged, stop scanning further
                    if nsfw_flag or violence_flag:
                        break

                frame_count += 1

            cap.release()

        else:
            # --- If image file ---
            nsfw_result = nsfw_detector(filepath)
            violence_result = violence_detector(filepath)

            if nsfw_result and nsfw_result[0]["label"].lower() == "nsfw":
                nsfw_flag = True

            bad_labels = ["assault", "weapon", "blood", "gore", "fight"]
            if violence_result:
                for res in violence_result:
                    if res["label"].lower() in bad_labels and res["score"] > 0.6:
                        violence_flag = True

        # --- Final Decision ---
        if nsfw_flag:
            os.remove(filepath)
            flash("❌ Nude/sexual content blocked!", "error")
            return redirect(request.url)

        if violence_flag:
            os.remove(filepath)
            flash("❌ Violent content blocked!", "error")
            return redirect(request.url)

        # Save metadata with username + time
        data = load_data()
        data[filename] = {
            "username": session["username"],
            "creator": creator,
            "description": description,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_data(data)

        flash("✅ File uploaded successfully!", "success")
        return redirect(url_for("index"))

    return render_template("upload.html")

# -------- Delete File --------
@app.route("/delete/<filename>")
def delete_file(filename):
    if "username" not in session:
        flash("⚠ Please login first!", "error")
        return redirect(url_for("login"))

    data = load_data()
    if filename in data and data[filename]["username"] == session["username"]:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        del data[filename]
        save_data(data)
        flash("✅ File deleted successfully!", "success")
    else:
        flash("❌ You can only delete your own uploads!", "error")

    return redirect(url_for("index"))

# ---------- START ---------- #
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")