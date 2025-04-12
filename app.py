from flask import Flask, render_template, request
#import backend as b

app = Flask(__name__)
#l = b.print_list()

@app.route("/", methods=["GET", "POST"])

def index():
    result = ""
    if request.method == "POST":
        name = request.form.get("name")
        result = f'Hello, {name}!'
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)