# VQA Demo
![](http://i.imgur.com/Q5pUnW4.png)

This demo is composed of two parts, a website and a background process.
When you have prepared both parts, you can start running.
```
$ python demo_att.py
```
(This process will wait for uploaded question-image pair)

in another terminal:
```
(django) $ python manage.py runserver
```
(This will start the web server, you can open the page (default: http://127.0.0.1/demo), and upload an image and a question.)

## Background Process
### Requirements
 * nltk
 * Caffe
 * Tensorflow
 * Gensim (optional; if you choose word2vec version)
    * word2vec model: ["*GoogleNews-vectors-negative300*"](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

We provide three versions of demo codes.
* *demo.py*  for [VQA-tensorflow](https://github.com/JamesChuanggg/VQA-tensorflow).
* *demo_att.py*  for our implementation of [san-vqa-tensorflow](https://github.com/TingAnChien/san-vqa-tensorflow)
* *demo_att_w2v.py*  is almost same as *demo_att.py*. It's a revision to use word2vec word embedding.

## Website
This website is built by [Django](https://www.djangoproject.com/).
If you are not familiar with it, you can follow the tutorials [here](https://tutorial.djangogirls.org/en/django_start_project/) to build your project.

The following codes are the components of this Django project.
Note that *upload.html* is provided in this folder.

settings.py  
```
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

urls.py
```
from django.conf.urls import include, url
from django.contrib import admin
from demo.views import upload_vqa
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^demo/$', upload_vqa),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

views.py
```
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseForbidden
from django.core.files.storage import FileSystemStorage
import json

def upload_vqa(request):
    fs = FileSystemStorage()
    uploaded_file_url = fs.url('gray.png') # default image
    att_image_url = fs.url('gray.png')     # default image
    if request.method == 'POST':
        if request.FILES['image'] and request.POST['question']:
            vqa_data = {}
            image = request.FILES['image']
            question = request.POST['question']
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            vqa_data['image'] = uploaded_file_url
            vqa_data['question'] = question
            # save uploaded question-image pair
            json.dump(vqa_data, open('vqa_data.json', 'w'))
            answer = ''
            # wait for processed result
            while 1:
                if fs.exists('vqa_ans.txt'):
                    with fs.open('vqa_ans.txt', 'r') as f:
                        answer = f.read()
                    att_image_url = fs.url('att.jpg')
                    fs.delete('vqa_ans.txt')
                    break
            return render(request, 'upload.html', {
                'uploaded_file_url': uploaded_file_url,
                'att_image_url': att_image_url,
                'input_question': question,
                'vqa_ans': answer
            })
    return render(request, 'upload.html', {
                'uploaded_file_url': uploaded_file_url,
                'att_image_url': att_image_url
            })
```



 



    
