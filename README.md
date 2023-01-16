#Predict API

Welcome to the documentation for the Predict API. This API allows you to predict the food item in a particular photo. Here is the api url: http://elliotfayman.pythonanywhere.com/

## Endpoints

The following endpoints are available:

- `GET /generate_key`: Retrieve a valid api token for valid users.
    - Generate_Key endpoint requires user to enter a username and valid password.
- `GET /predict`: Predicts food item in image and returns result as JSON object.
    - Predict endpoint has two paramaters, token and image.
    - Enter the token generated using the `GET /generate_key` endpoint
    - Enter the image url for the image you would like to classify


## Authentication

To access the API, you will need to provide an API key. This key can be obtained by generated using the `GET /generate_key` endpoint

## Examples

Here are some examples of how to use the API:

Ruby Example:

```
    HTTParty.get('http://elliotfayman.pythonanywhere.com/predict?token=tokenKey&image=https://th.bing.com/th/id/R.5c487ffb0b1b3e854764a1e9bfd43ff2?rik=NAgVDCH4nrhSzg&pid=ImgRaw&r=0')

```


## Error handling

If an error occurs, the API will return a JSON object with an `error` field containing a description of the error.

## Additional information

For more information on how to use the API, contact me at elliotfayman@gmail.com
