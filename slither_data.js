// ==UserScript==
// @name         Slither data collection
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        http://slither.io/
// @icon         https://www.google.com/s2/favicons?sz=64&domain=slither.io
// @grant        none
// ==/UserScript==

/*
    * copyright (c) 2022 Aaron Hance
    * slither_data.js is freely distributable under the MIT license.
    * http://www.opensource.org/licenses/mit-license.php
    *
    * ------------------------------------------------------------
    *
    * Script for recording slither.io games for ai training.
    * saves input, score and screnshot data to a file.
    *
    * ------------------------------------------------------------
    *
    * Usage:
    * 1. Open slither.io game.
    * 2. Run script.
    * 3. use arrow keys to control slither.
    *
*/

//key states
var key_states = {
    left: false,
    right: false,
    up: false
}

//find canvas that has a width greater than 0
var canvas = document.getElementsByTagName('canvas');
for (var i = 0; i < canvas.length; i++) {
    if (canvas[i].width > 0) {
        canvas = canvas[i];
        break;
    }
}

function get_score() {
    //find element containing "Your length:", search for it
    var xpath = "//*[contains(text(), 'Your length:')]";
    var score = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    score = score.parentElement;
    //get last child
    score = score.lastChild.innerText;

    return score;
}

function get_screen() {
    // var data = canvas.toDataURL();
    // console.log(data);
    // //remove "data:image/png;base64," from string
    // data = data.substring(data.indexOf(",") + 1);

    var dsnakes = window.snakes;
    var dfoods = window.foods;
    var dsnake = window.snake;

    //for each snake get 'pts' 'dead_amt'
    var snakes = [];
    for (var i = 0; i < dsnakes.length; i++) {
        var snake = {
            pts: dsnakes[i].pts,
            dead_amt: dsnakes[i].dead_amt
        }
        snakes.push(snake);
    }

    var snake = {   
        pts : dsnake.pts,
        dead_amt : dsnake.dead_amt,
        xx : dsnake.xx,
        yy : dsnake.yy,
    }

    //for each food get 'xx' 'yy'
    var foods = [];
    for (var i = 0; i < dfoods.length; i++) {

        if (dfoods[i] == null) {
            continue;
        }

        var food = {
            xx: dfoods[i].xx,
            yy: dfoods[i].yy
        }
        foods.push(food);
    }

    var data = JSON.stringify([snakes, foods, snake]);

    return data;
}

function get_is_playing() {
    return window.playing;
}

function key_lsitener() {
    document.addEventListener('keydown', function (e) {

        console.log("keydown");
        console.log(key_states)

        if (e.which == 37) {
            console.log("left");
            key_states.left = true;
        }
        if (e.which == 38) {
            console.log("up");
            key_states.up = true;
        }
        if (e.which == 39) {
            console.log("right");
            key_states.right = true;
        }
    });
    document.addEventListener('keyup', function (e) {
        if (e.which == 37) {
            key_states.left = false;
        }
        if (e.which == 38) {
            key_states.up = false;
        }
        if (e.which == 39) {
            key_states.right = false;
        }
    });
}

var gdata = [];

function loop() {

    // console.log("loop");

    if (get_is_playing()) {
        var score = get_score();
        var screen = get_screen();
        var keys = JSON.stringify(key_states);
        var sample = {
            score: score,
            screen: screen,
            keys: keys
        }
        gdata.push(sample);
    }
    else if (gdata.length > 0) {
        console.log(gdata);
        gdata = [JSON.stringify(gdata)];
        //blob data and download
        var blob = new Blob(gdata, {
            type: "application/json"
        });

        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = "slither_data.json";
        a.click();
        URL.revokeObjectURL(url);

        gdata = [];
    }
}

key_lsitener();
setInterval(loop, 80);