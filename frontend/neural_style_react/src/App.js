import './App.css';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.css';
import React, { useEffect, useState, useRef } from "react";

function App() {

  const postURL = 'http://127.0.0.1:8000/uploadfiles';

  const [myImage, setMyImage] = useState([]);
  const [neuralImage, setneuralImage] = useState();

  // const addImage = e => {
  //   const nowSelectImageList = e.target.files;
  //   const nowImageURLList = [];
  //   for (let i = 0; i < nowSelectImageList.length; i += 1) {
  //     const nowImageUrl = URL.createObjectURL(nowSelectImageList[i]);
  //     nowImageURLList.push(nowImageUrl);
  //   }
  //   console.log(nowImageURLList);
  //   setMyImage(nowImageURLList);
  // }

  let myImages = (images) => {
    return images.map((image) => {
      return <img src={image}></img>
    })
  }

  const onSubmit = async (e) => {
    e.preventDefault();
    e.persist();

    let files = e.target.profile_files.files;
    let formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    // console.log(formData);

    const postSurvey = await axios({
      method: 'POST',
      url: 'http://127.0.0.1:8000/uploadfiles',
      // mode: "cors",
      headers: {
        "Content-Type": "multipart/form-data",
      },
      data: formData
    });
    // console.log(postSurvey)
    // console.log(formData)
  };

  return (
    <>
      <form onSubmit={(e) => onSubmit(e)}>
        <input
          type="file"
          name="profile_files"
          multiple="multiple"
        />

        <button type="submit">제출</button>
      </form>
    </>
  );
}

export default App;
