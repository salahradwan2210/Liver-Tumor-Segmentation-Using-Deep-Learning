import sre_compile
from flask import Flask, render_template, request, redirect, jsonify, url_for, flash, session
import sqlite3
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
import os
from fastai.vision.all import * 
import csv

app = Flask(__name__)
db_local = 'patients.db'
global ltr, lv, sr  # Declare them outside any function
result=""
ltr = "-"
lv = 0
sr = "-"

path = Path(os.getcwd())
def get_x(fname:Path): return fname
def label_func(x): return path/'./train_masks/train_masks/'
def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean() 

def cust_foreground_acc(inp, targ):  # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) # 3 is a dummy value to include the background which is 0

learn0 = keras.models.load_model("./unet_model (2).h5" )

UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

result = ""
ses = False
name = ""



def malignantBeningCheck(count):
    if(count <500):
        return 'malignant'
    else:
        return 'Benign'

def process_img(img, add_pixels_value=0):
    new_img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    return np.array(new_img, dtype=object)

def predict_mask(model, img_path):
    img = cv2.imread(img_path)
    processed_img = process_img(img)

    # Expand dimensions if needed for your model input shape
    img_batch = np.expand_dims(processed_img, axis=0)

    try:
        # Perform prediction using your model
        preds = model.predict(img_batch)
        print(f"Model predictions: {preds}")  # Debugging: Print the raw predictions
        predicted_mask = np.asarray(preds[0][0], dtype=np.uint8)
        return predicted_mask
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def check(number):
    print(f"Check function received number: {number}")  # Debugging: Print the received number
    if number == 0:
        return 'Tumor Not Detected'
    elif number == 1:
        return 'Tumor Detected'
    else:
        return 'Tumor Detected'
    

def process_img(img, add_pixels_value=0):
    new_img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

    return np.array(new_img, dtype=object)

def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

dicom_windows = types.SimpleNamespace(
    brain=(80,40),
    subdural=(254,100),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400),
    custom = (200,60)
)

@patch
def windowed(self:Tensor, w, l):
    px = self.clone()
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)

class TensorCTScan(TensorImageBW): _show_args = {'cmap':'bone'}

@patch
def freqhist_bins(self:Tensor, n_bins=100):
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()
    
@patch
def hist_scaled(self:Tensor, brks=None):
    if self.device.type=='cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks)) 
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.,1.)
    
    
@patch
def to_nchan(x:Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins,int) or bins!=0: res.append(x.hist_scaled(bins).clamp(0,1))
    dim = [0,1][x.dim()==3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch
def save_jpg(x:(Tensor), path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins)*255).byte()
    im = Image.fromarray(x.permute(1,2,0).numpy(), mode=['RGB','CMYK'][x.shape[0]==4])
    im.save(fn, quality=quality)

def processNiifiles(filepath):
    curr_ct        = read_nii("static/"+filepath)
    curr_file_name = str(filepath).split('.')[0]
    curr_dim       = curr_ct.shape[2] # 512, 512, curr_dim
    curr_count = 0
    file_array = []
    for curr_slice in range(int(curr_dim/2),curr_dim,3): 
        data = tensor(curr_ct[...,curr_slice].astype(np.float32))
        curr_count += 1
        save_file_name = f'{curr_file_name}_slice_{curr_count}.jpg'
        data.save_jpg(f"static/"+save_file_name, [dicom_windows.liver,dicom_windows.custom])
        file_array.append(save_file_name)
        if(curr_count == 4):
            break;
    return file_array

@app.route("/nii", methods=["GET", "POST"])
def nii():
        

    if request.method == 'POST':
        for filename in os.listdir('static/'):
            if filename.startswith('work'):  # not to remove other images
                os.remove('static/' + filename)
            if filename.startswith('scan'):
                os.remove('static/' + filename)
            if filename.startswith('pred'):
                os.remove('./static/' + filename)
            if filename.startswith('slice'):
                os.remove('static/' + filename)
                
        file = request.files['nii']
        if 'nii' not in request.files:
            return render_template('nii.html', ses=ses, error="File not found!!!")

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('nii.html', ses=ses, error="File not found!!!")
        if file:
            filename = secure_filename(file.filename)
            timeStamp ="work" + str(time.time()) + ".nii"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timeStamp )
            
            file.save(filepath)
            file_array = processNiifiles(timeStamp)

            #print(prediction)
            return render_template("imagesNii.html",file_array = file_array, ses=ses,name=name, error="")
    return render_template("nii.html",name= name, ses=ses, error="")

@app.route("/imagesNii", methods=['GET', 'POST'])
def imagesNii():
    return render_template('imagesNii.html', ses=ses, error="")

@app.route("/predNii/<id>")
def predNii(id):
    timeStamp = None
    found = False

    # Search for the file with the given ID in the static directory
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if f'slice_{id}' in filename:
            timeStamp = filename
            found = True
            break

    # If no file is found with the ID, create a new timestamped file path
    if not found:
        timeStamp = "work" + str(time.time()) + ".jpg"
    
    # Construct the full file path
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], timeStamp)

    # Check if the file exists before processing
    if not os.path.exists(filepath):
        return "File not found", 404

    # Process the image using Pillow
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((128, 128))  # Resize to the input size expected by the model
        img_array = np.expand_dims(np.array(img), axis=0) / 255.0  # Normalize the image
    except Exception as e:
        return f"Error processing image: {e}", 500

# Predict using the model (assuming learn0 is your model)
    predicted_mask = learn0.predict(img_array)[0]
    threshold = 0.5

    predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)

    # Save the predicted mask using matplotlib
    pred_path = "pred" + str(time.time()) + ".png"
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], pred_path), predicted_mask_binary[:, :, 0], cmap='gray')

    # Check if the predicted mask is black only
    pred_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], pred_path)).convert('L')
    pred_array = np.array(pred_img)
    if np.all(pred_array == 0):
        number = 0
        
    else:
        number = 1

    print(f"Classification: {number}")  # Debugging: Print the classification result
    predicted_results = check(number)
    print(f"Predicted Results: {predicted_results}")  # Debugging: Print the predicted results
    result = 1
    unique, counts = np.unique(predicted_results, return_counts=True)
    pred_matrix = np.array((unique, counts)).T
    print(pred_matrix)
    
    liver_visiblity = 0
    liver_tumor_ratio = 0
    size_result = "-"

    if 1 in unique:
        liver_visiblity = pred_matrix[1][1] / (pred_matrix[0][1] + pred_matrix[1][1]) * 100
        liver_visiblity = float("{:.2f}".format(liver_visiblity))
        print(f"Liver visibility: {liver_visiblity}")  # Debugging: Print the liver visibility
        print(f"Liver tumor ratio: {liver_tumor_ratio}")  # Debugging: Print the liver tumor ratio
        print(f"Size result: {size_result}")  # Debugging: Print the size result
    # tumor_ratio = calculate_tumor_ratio(predicted_mask, img_array)
    if number == 0 :
        tumor_ratio = 0 
        formatted_tumor_ratio = round(tumor_ratio, 2)

        tumor_type = "The patient is healthy"
    else:
        tumor_ratio = calculate_tumor_ratio(predicted_mask, img_array)
        formatted_tumor_ratio = round(tumor_ratio, 2)

        tumor_type = classify_tumor_type(predicted_mask, img_array)

    global ltr,lv, sr
    ltr = liver_tumor_ratio
    lv = liver_visiblity
    sr = size_result



            #print(prediction)
    return render_template("result.html", img1 = timeStamp,  img3 = pred_path,classification=number, predicted_results=predicted_results,liver_visiblity=formatted_tumor_ratio, liver_tumor_ratio=liver_tumor_ratio, size_result=tumor_type,ses=ses,name=name, error="")



def calculate_tumor_ratio(tumor_mask, organ_mask):
    tumor_area = np.sum(tumor_mask)
    organ_area = np.sum(organ_mask)
    
    if organ_area == 0:
        return 0  # To avoid division by zero if organ_mask is somehow empty
    
    tumor_ratio = ((tumor_area / organ_area) * 100)
    return tumor_ratio
def classify_tumor_type(tumor_mask, organ_mask):
    # Example criteria for tumor type based on size
    tumor_area = np.sum(tumor_mask)
    organ_area = np.sum(organ_mask)
    tumor_ratio = ((tumor_area / organ_area) * 100)

    if tumor_ratio > 0 and tumor_ratio < 3:
        return "Small benign tumor"
    elif tumor_ratio > 3 and tumor_ratio< 10:
        return "Medium benign tumor"
    elif tumor_ratio > 10 and tumor_ratio < 30:
        return "Large benign tumor"
    else:
        return "Potentially malignant tumor"

@app.route("/mainPage", methods=["GET", "POST"])
def mainPage():


    
    if request.method == 'POST':
        # Handle file upload
        for filename in os.listdir('static/'):
            if filename.startswith('work') or filename.startswith('scan') or filename.startswith('pred') or filename.startswith('slice'):
                os.remove('static/' + filename)

        file = request.files['mri']
        if 'mri' not in request.files or file.filename == '':
            return render_template('index.html', ses=ses, error="File not found!!!")

        filename = secure_filename(file.filename)
        timeStamp = "work" + str(time.time()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timeStamp)
        file.save(filepath)
            # Process the image to get masks (this part depends on your specific implementation)
        # tumor_mask, organ_mask = process_image(filepath)

        # Calculate the tumor ratio

        # Process the image for prediction using Pillow
        img = Image.open(filepath)
        img = img.resize((128, 128))  # Resize to the input size expected by the model
        img_array = np.expand_dims(np.array(img), axis=0) / 255.0  # Normalize the image
        img_path0 = "scan" + str(time.time()) + ".jpg"

        # Predict using the model
        predicted_mask = learn0.predict(img_array)[0]
        
        print(f"Predicted mask: {predicted_mask}")  # Debugging: Print the predicted mask
        threshold = 0.5  # Adjust this threshold based on your model's output
        predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)

        # Save the predicted mask using matplotlib
        pred_path = "pred" + str(time.time()) + ".png"
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], pred_path), predicted_mask_binary[:, :, 0], cmap='gray')

        # Check if the predicted mask is black only
        pred_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], pred_path)).convert('L')
        pred_array = np.array(pred_img)
        if np.all(pred_array == 0):
            number = 0
            
        else:
            number = 1

        print(f"Classification: {number}")  # Debugging: Print the classification result
        predicted_results = check(number)
        print(f"Predicted Results: {predicted_results}")  # Debugging: Print the predicted results
        result = 1
        unique, counts = np.unique(predicted_results, return_counts=True)
        pred_matrix = np.array((unique, counts)).T
        print(pred_matrix)
        
        liver_visiblity = 0
        liver_tumor_ratio = 0
        size_result = "-"

        if 1 in unique:
            liver_visiblity = pred_matrix[1][1] / (pred_matrix[0][1] + pred_matrix[1][1]) * 100
            liver_visiblity = float("{:.2f}".format(liver_visiblity))
            print(f"Liver visibility: {liver_visiblity}")  # Debugging: Print the liver visibility
            print(f"Liver tumor ratio: {liver_tumor_ratio}")  # Debugging: Print the liver tumor ratio
            print(f"Size result: {size_result}")  # Debugging: Print the size result
        # tumor_ratio = calculate_tumor_ratio(predicted_mask, img_array)
        if number == 0 :
            tumor_ratio = 0 
            formatted_tumor_ratio = round(tumor_ratio, 2)

            tumor_type = "The patient is healthy"
        else:
            tumor_ratio = calculate_tumor_ratio(predicted_mask, img_array)
            formatted_tumor_ratio = round(tumor_ratio, 2)

            tumor_type = classify_tumor_type(predicted_mask, img_array)



        global ltr, lv, sr 
        lv = liver_tumor_ratio
        sr = tumor_type     
        ltr = formatted_tumor_ratio

        return render_template("result.html", img1=timeStamp, img2=img_path0, img3=pred_path, classification=number, predicted_results=predicted_results, liver_visiblity=formatted_tumor_ratio, liver_tumor_ratio=liver_tumor_ratio, size_result=tumor_type)



    return render_template("index.html", name=name, ses=ses, error="")

#  ============================================== form data backend ==============================================

@ app.route("/form",  methods=["GET", "POST"])
def form():

    if request.method == "POST":
        user_details = (
            request.form['name'],
            request.form['age'],
            request.form['gender'],
            request.form['bgrp'],
            request.form['mHist'],
            request.form['pNo'],
            request.form['tdate'],
            request.form['report']
        )
        insertdata(user_details)
        user_data = query_data()
        dr_data = query_dr_data(name)
        
        return render_template('display.html',user_data=user_data,dr_data = dr_data, ltr =ltr,  sr=sr, ses=ses, name=name)
    return render_template('info.html', ses=ses, name =name)


def insertdata(user_details):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    sql_execute_string = 'INSERT INTO pInfo(pname, page, pgender, pbgrp, pmedhist, pphone, pdate, presult) VALUES (?,?,?,?,?,?,?,?)'
    c.execute(sql_execute_string, user_details)
    conn.commit()
    conn.close()

    # Save patient data to a CSV file
    with open('patients.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(user_details)

def query_dr_data(name):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    c.execute("select * from logindb where username=?",
                  (name,))
    dr_data = c.fetchall()
    return dr_data

def query_data():
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    c.execute("""
       SELECT * 
    FROM    pInfo
    WHERE   id = (SELECT MAX(id)  FROM pInfo);

    """)
    user_data = c.fetchall()
    return user_data

@app.route("/displayData", methods=["GET", "POST"])
def displayData():
    if request.method == "GET":
        user_data = query_data()
        dr_data = query_dr_data(name)
        return render_template('display.html',user_data=user_data,dr_data = dr_data,  lv=lv, sr=sr,ltr =ltr, ses=ses, name=name)

#  ============================================== Login/Sign Up Backend ==============================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":
        conn = sqlite3.connect(db_local)
        c = conn.cursor()
        username = request.form['username']
        password = request.form['password']
        fullname = request.form['fullname']
        emailid = request.form['emailid']
        hname = request.form['hname']
        position = request.form['position']

        c.execute("SELECT * FROM logindb WHERE username = ?", [username])
        if c.fetchone() is not None:
            return render_template('signup.html', error="Username already taken")
        elif(checkpass(password) == True):
            c.execute(
                "INSERT INTO logindb(username, password, fullname, emailid, hname, position ) VALUES (?,?,?,?,?,?)", 
                (username, password, fullname, emailid, hname, position))
            conn.commit()
            conn.close()

            # Save the new user details to a CSV file
            with open('patients.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([username, password, fullname, emailid, hname, position])

            return redirect(url_for('login'))
        else:
            return render_template('signup.html', ses=ses, error="The password is weak, use another!!!")
        conn.commit()
        conn.close()
        return render_template('signup.html', ses=ses, error="")
    else:
        return render_template('signup.html', ses=ses, error="")


def checkpass(password):
    Special = ['$', '@', '#']
    if len(password) < 8 or len(password) > 15:
        return False
    if not any(char.isdigit() for char in password):
        return False
    if not any(char.isupper() for char in password):
        return False
    if not any(char.islower() for char in password):
        return False
    if not any(char in Special for char in password):
        return False
    return True


@ app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = sqlite3.connect(db_local)
        c = conn.cursor()
        username = request.form['username']
        password = request.form['password']
        c.execute("select * from logindb where username=? and password =?",
                  (username, password))
        row = c.fetchone()
        if row == None:
            return render_template('login.html', error="Login Failed: No such user exists")
        else:
            global ses
            global name
            ses = True
            name = row[0]
            return render_template('index.html',name = name, ses=ses, error="")

    return render_template("login.html", error="")


@ app.route("/logout")
def logout():
    global ses
    ses = False
    name = ''

    return render_template('login.html', ses=ses, name=name, error='')


#================================== DOCTOR'S PAGE ==============================================

@ app.route("/doctors/<city>")
def doctors(city):
    conn = sqlite3.connect(db_local)
    c = conn.cursor()
    city = str(city)
    c.execute("select * from doctors where city=? ",
              (city,))
    data = c.fetchall()
    conn.close()
    return render_template('doctors.html', data=data, error="")


if __name__ == '__main__':


    app.run(debug=True)
    app.run(debug=True)
