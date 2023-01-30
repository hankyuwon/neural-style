import './App.css';
import axios from 'axios';
// import 'bootstrap/dist/css/bootstrap.css';
import React, { useEffect, useState, useRef } from "react";

function App() {

  let [neural, setNeural] = useState(null);

  const fuckTest = async () => {
    const fuckReturn = await axios({
      method: 'POST',
      url: 'http://127.0.0.1:8000/fucks'
    })
    console.log(fuckReturn.data);
  }

  const onSubmit = async (e) => {
    e.preventDefault();
    e.persist();

    let files = e.target.profile_files.files;
    let formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    const postSurvey = await axios({
      method: 'POST',
      url: 'http://127.0.0.1:8000/uploadfiles',
      //url: 'http://127.0.0.1:8000/image_upload',
      mode: "cors",
      headers: {
        "Content-Type": "multipart/form-data",
      },
      data: formData,
      responseType: 'blob'
    });
    const url = window.URL.createObjectURL(postSurvey.data);
    setNeural(url);
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
      <img src={`${neural}`}></img>
    </>
  );
}

export default App;
