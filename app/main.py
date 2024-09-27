from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ml_yolo import ml_yolo
import base64
import os


yolo = ml_yolo()

app = FastAPI()

# Configure CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



@app.get("/test/")
def test_endpoint():
    """
    Test endpoint to verify the API is working.

    Returns:
        JSONResponse: A JSON response indicating the API is running.
    """
    return JSONResponse(content={"message": "API is working!"})


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint for uploading an image.

    Args:
        image (UploadFile): The image file to be uploaded.

    Returns:
        JSONResponse: A JSON response containing the detected inventory and encoded image.
    """

    file_location = f"images/{image.filename}"
    
    # Save the uploaded image to the file system
    with open(file_location, "wb") as f:
        content = await image.read()  
        f.write(content)  

    # Process the uploaded image using the YOLO model
    inventory_df = yolo.process_image(file_location)
    
    # Read the saved image file to encode it in base64 format
    with open(file_location, "rb") as f:
            image_data = f.read() 

    encoded_image = base64.b64encode(image_data).decode('utf-8')
    os.remove(file_location)

    # Check if the processing resulted in a valid inventory DataFrame
    if inventory_df is not None and not inventory_df.empty:
        # Convert the DataFrame to a list of dictionaries with 'Item' and 'Count'
        inventory_list = inventory_df[['Item', 'Count']].to_dict(orient='records')
        # Filter out unwanted items
        inventory_list = [item for item in inventory_list if item['Item'] != 'refrigerator']
    
        # Prepare response object with inventory and image
        # Example inventory_list: [{'Item': 'apple', 'Count': 2}, {'Item': 'orange', 'Count': 1}]
        obj =  {"inventory": inventory_list, "image": encoded_image}
        return JSONResponse(content=obj)

    # If no valid inventory, return just the encoded image
    obj =  {"image": encoded_image}
    return JSONResponse(content=obj)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)