from distutils import debug
from flask import Flask


app = Flask(__name__)
#In memory database
posts = {
    0: {
        'title': 'Hello, World!',
        'content': 'This is my first blog post!'
    }    
}

#URL Routing
@app.route('/')
def home():
    return "Hello, World!"

@app.route('/post/<int:post_id>')
def post(post_id):
    post = posts.get(post_id)
    return f"Post {post['title']}, content: \n\n{post['content']}"


#Runs the app
if __name__ == '__main__':
    app.run(debug=True)
    