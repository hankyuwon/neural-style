# Neural Style & React

React & Fast API 방학 프로젝트 (2023-01-28 ~ 2023-01-29)

기존에 존재하는 Neural Style Model 에서 API 통신이 가능하게 약간의 변형 후 React 웹에서 Content, Style Image 를 제출하면 Neural_Style Image를 받아볼 수 있다.


## Example 1

<figure class = "third">
   Content Image
<img src = "https://github.com/hankyuwon/neural-style/blob/main/backend/examples/gitimage/1.jpg" width = "400" height="300"/>
  
   Style Image
  
<img src = "https://github.com/hankyuwon/neural-style/blob/main/backend/examples/gitimage/2.jpg" width = "400" height="300"/>
  
   Neural Image
  
<img src = "https://github.com/hankyuwon/neural-style/blob/main/backend/examples/gitimage/3.jpg" width = "400" height="300"/>
</figure>

## Example 2
<figure class = "third">
  Content Image
<img src = "https://github.com/hankyuwon/neural-style/blob/main/backend/examples/gitimage/4.png" width = "400" height="300"/>
  
  Style Image
  
<img src = "https://github.com/hankyuwon/neural-style/blob/main/backend/examples/gitimage/5.jpg" width = "400" height="300"/>
  
  Neural Image
  
<img src = "https://github.com/hankyuwon/neural-style/blob/main/backend/examples/gitimage/6.jpg" width = "400" height="300"/>
</figure>


### SETTING

##### 1. create conda environments
```bash
conda create -n "envname"
```

##### 2. move directory
```bash
 cd /neural_style/backend
 ```
  ##### 2-1. install package
```bash
$ pip install -r requirements.txt
$ pip install fastapi
$ pip install uvicorn
```

##### 3. run fastapi

```bash
$ uvicorn backend:app --reload
```

##### 4. move directory (run React)

```bash
 cd /frontend/neural_style_react
```
```bash
$ npm start
```

##### 5. Select content Image, Style Image and submit



</br></br></br>
## License

Copyright (c) 2015-2021 Anish Athalye. Released under GPLv3. See
[LICENSE.txt][license] for details.

[net]: https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[adam]: http://arxiv.org/abs/1412.6980
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[lengstrom-fast-style-transfer]: https://github.com/lengstrom/fast-style-transfer
[fast-neural-style]: https://arxiv.org/pdf/1603.08155v1.pdf
[license]: LICENSE.txt
