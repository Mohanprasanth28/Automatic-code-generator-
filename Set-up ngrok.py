from flask import Flask
from pyngrok import ngrok
ngrok.set_auth_token(" '''set your ngrok token''' ")
public_url=ngrok.connect(5000).public_url
