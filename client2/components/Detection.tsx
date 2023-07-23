import React, { useEffect, useState } from "react";
import axios from "axios";
import Header from "./Header";
import Image from "next/image";
import toast from "react-hot-toast";

const Detection = () => {
  const [image, setImage] = useState<any>("");
  const [imagePreview, setImagePreview] = useState<any>("");
  const [detection, setDetection] = useState<string>("");
  const [text, setText] = useState<boolean>(true);

  useEffect(() => {
    if (!image) {
      setImagePreview(undefined);
      return;
    }
    const objUrl = URL.createObjectURL(image);
    setImagePreview(objUrl);
    return () => URL.revokeObjectURL(objUrl);
  }, [image]);

  const handleDetections = async (e: any) => {
    e.preventDefault();
    setText(false);
    const id = toast.loading("Detecting...");
    try {
      const formData = new FormData();
      formData.append("file", image);
      const res = await axios.post("http://127.0.0.1:8000/predict", formData);

      toast.success("Detection Successfull", {
        id: id,
      });
      setDetection(res.data);
      console.log(res);
    } catch (error) {
      toast.error("Detection Failed!!", {
        id: id,
      });
      console.log(error);
    }
  };

  return (
    <div>
      <Header />
      <div className="flex flex-col md:flex-row items-center justify-center md:mt-32 mt-16 space-y-5 md:space-x-40 ">
        <div>
          <div className="h-80 w-80 rounded-md border-2 border-black border-dashed">
            {image ? (
              <img
                src={imagePreview}
                alt={image.name}
                className="h-full w-full rounded-md"
              />
            ) : (
              // <Image src={image} alt={image.name} height={320} width={320} />
              <p className="text-xs text-gray-400 p-2 text-center mt-36">
                (Select images related to building, sea, glacier, mountain &
                forest only, as the model is trained for above classes only.)
              </p>
            )}
          </div>
          <div className="flex flex-col items-center justify-center space-y-4">
            <input
              type="file"
              name="file"
              onChange={(e: any) => setImage(e.target.files[0])}
              className="mt-3"
            />
            <button
              disabled={!image}
              onClick={handleDetections}
              className="bg-red-600 text-white py-1 px-2 rounded-sm hover:bg-red-400"
            >
              Detect
            </button>
          </div>
        </div>
        <div className="">
          {detection.length > 0 && (
            <p className="text-xs w-40 p-2 text-center">
              <span className="text-md">Detected image is:</span> <br />
              <span className="text-3xl font-bold">{detection}</span>
            </p>
          )}
          {text && (
            <p className="text-xs w-40 p-2 text-center">
              (Click detect after selecting image related to specified classes
              to get predictions.Image formats png , jpg & jpeg are preferred.)
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Detection;
