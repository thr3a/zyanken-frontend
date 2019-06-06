const CLASSES = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four',5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}

window.onload = function () {
	loadModel();
};

let model;
async function loadModel() {
	console.log("model loading..");
	$("#console").html(`<li>model loading...</li>`);
	model = await tf.loadLayersModel(`${location.protocol}//${location.host}/models/model.json`);
	console.log("model loaded.");
	$("#console").html(`<li>VGG16 pre trained model loaded.</li>`);
	startWebcam();
	const url = new URL(location.href);
	const interval = parseInt(url.searchParams.get("i")) || 300;
	setInterval(predict, interval);
};

var video;
function startWebcam() {
	console.log("video streaming start.");
	$("#console").html(`<li>video streaming start.</li>`);
	video = $('#main-stream-video').get(0);
	vendorUrl = window.URL || window.webkitURL;

	navigator.getMedia = navigator.getUserMedia ||
						 navigator.webkitGetUserMedia ||
						 navigator.mozGetUserMedia ||
						 navigator.msGetUserMedia;

	navigator.getMedia({
		video: true,
		audio: false
	}, function(stream) {
		localStream = stream;
		video.srcObject = stream;
		video.play();
	}, function(error) {
		alert("Something wrong with webcam!");
	});
}

async function predict(){
	let tensor = captureWebcam();

	let prediction = await model.predict(tensor).data();
	let results = Array.from(prediction)
				.map(function(p,i){
	return {
		probability: p,
		className: CLASSES[i]
	};
	}).sort(function(a,b){
		return b.probability-a.probability;
	}).slice(0,4);

	$("#console").empty();
	if (results[0].className == 'zero') {
		var hand = 'グー';
	} else if (results[0].className == 'one' || results[0].className == 'two' || results[0].className == 'three' || results[0].className == 'six' ) {
		var hand = 'チョキ';
	} else {
		var hand = 'パー';
	}
	$("#console").text(`${hand} ${results[0].probability.toFixed(2)}`);
	let msg = '';
	results.forEach(function(p){
		// $("#console").append(`<li>${p.className} : ${p.probability.toFixed(6)}</li>`);
		msg += `${p.className}: ${p.probability.toFixed(2)} `;
	});
	console.log(msg);
};

function captureWebcam() {
	var canvas    = document.getElementById('mycanvas');
	var context   = canvas.getContext('2d');
	context.drawImage(video, 0, 0, 250, 250, 0, 0, 250, 250);
	tensor_image = preprocessImage(canvas);

	return tensor_image;
}

function preprocessImage(image){
	let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([100,100]).toFloat();	
	let offset = tf.scalar(255);
  return tensor.div(offset).expandDims();
}