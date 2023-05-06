from time import time
st = time()
from flask import Flask,request,render_template,send_from_directory
import gen_object
#import generate_images
import os
import shutil
UPLOAD_FOLDER = os.path.join(os.getcwd(),'backgrounds')
app = Flask(__name__, static_folder=os.path.join(os.getcwd(),"static"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_SC'] = os.path.join(os.getcwd(),'source')
end = time()
print('Seconds for initialization of the whole app :',end-st)

@app.route('/',methods=['GET','POST'])
def home():
    logo_path = os.path.join(os.getcwd(),'josh_logo.png')
    if request.method=='GET':
        return render_template('home.html',logo_path=logo_path)
    elif request.method=='POST':
        st = time()
        source_path = request.form.get('source_path')
        #num_img = int(request.form.get('car_num'))

        #below saving the images using which the dataset is being generated
        bg_img = request.files['bg_file']
        bg_img.save(os.path.join(app.config['UPLOAD_FOLDER'],bg_img.filename))
        sc_img = request.files['sc_file']
        sc_img.save(os.path.join(app.config['UPLOAD_FOLDER_SC'],sc_img.filename))

        #obj = generate_images.ImageGenerator()
        #obj.generate_source(num_img)
        #print('Images of cars generated')
        # Two lines above utilize the GAN for generating images of the cars. When needed uncomment above 2 lines and line where
        # variable num_img is created, (line 18)

        # Below is the part of the code which takes the longest time to execute. Here the segmentation model is segmenting out cars from the 
        # Source images provided, any optimization to the inference go to gen_object.py 
        source_path=os.path.join(os.getcwd(),'source')
        background = os.path.join(os.getcwd(),'backgrounds')
        print('Started Dataset Generation')
        obj2 = gen_object.DatasetGenerate(source_path,background,int(request.form.get('car_num')),int(request.form.get('max_objects')))
        result_imgs, bbox = obj2.build_dataset()
        print('Images for dataset generated')
        ans=""
        for i in bbox:
            ans=ans+','.join([str(x) for x in i])+'\n'

        # Now here generating the bounding box CSV file and creating the compressed folder with the images of the dataset.
        print('Building the CSV')
        with open('bounding_box.csv','w') as f:
            f.write(ans)
        print('Done with Building the CSV')
        shutil.make_archive('dataset_images_compressed','zip','dataset_images')


        # Now below part removes the files once the dataset has been generated
        folder_path = os.path.join(os.getcwd(),'backgrounds')
        for file1 in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file1)
            os.remove(file_path)
        
        # Uncomment the below 3 lines if you want to remove the source images which were uploaded earlier by the user.
        folder_path = os.path.join(os.getcwd(),'source')
        for file2 in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file2)

        folder_path = os.path.join(os.getcwd(),'dataset_images')
        for file3 in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file3)
        end = time()
        print('=================================================')
        print('Time taken to handle on dataset generation request :',end-st)
        print('**************************************************************')
        return render_template('home.html',data_generated=True,logo_path=logo_path)

# This route handles uploading of the background from user end (Not used now as a single form handles all)
# @app.route('/background_upload',methods=['POST'])
# def background():
#     logo_path = os.path.join(os.getcwd(),'josh_logo.png')
#     if request.method=='POST':
#         file = request.files['bg_file']
#         if file:
#             filename = file.filename
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
#     return render_template('home.html',logo_path=logo_path)

# This route handles uploading of the car image from user end (Not used now as a single form handles all)
# @app.route('/source_upload',methods=['POST'])
# def source():
#     logo_path = os.path.join(os.getcwd(),'josh_logo.png')
#     if request.method=='POST':
#         file = request.files['sc_file']
#         if file:
#             filename = file.filename
#             file.save(os.path.join(app.config['UPLOAD_FOLDER_SC'],filename))
#     return render_template('home.html',logo_path=logo_path)

# The 2 routes below are for handling the download of bounding_box co-ordinates and dataset images.
@app.route('/uploads1', methods=['GET','POST'])
def download_bbox():
    return send_from_directory(os.getcwd(),'bounding_box.csv')

@app.route('/uploads22',methods=['POST','GET'])
def download_dataset():
    return send_from_directory(os.getcwd(),'dataset_images_compressed.zip')

if __name__=='__main__':
    app.run(debug=True)