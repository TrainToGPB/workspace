from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

import uvicorn

app = FastAPI()


@app.post("/files/") # /files로 POST하는 경우 아래 함수 호출
def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]} # 여기는 file size 반환


@app.post("/uploadfiles/") # /uploadfiles로 POST하는 경우 아래 함수 호출
def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]} # 여기는 file name 반환


@app.get("/")
def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)


