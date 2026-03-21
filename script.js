const API = "https://saferoute-backend-s5cm.onrender.com";

var map = L.map('map').setView([13.0827, 80.2707], 13);

L.tileLayer(
'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
{
maxZoom: 19
}).addTo(map);

var marker = null;
var clickedLat = null;
var clickedLng = null;

map.on('click', function(e){

clickedLat = e.latlng.lat;
clickedLng = e.latlng.lng;

if(marker){
map.removeLayer(marker);
}

marker = L.marker([clickedLat, clickedLng]).addTo(map);

});

async function checkRisk(){

if(clickedLat == null){
alert("Click on map first");
return;
}

const data = {
temperature:30,
humidity:70,
visibility:2,
wind_speed:10,
weather:3,
Junction:1,
traffic_signal:0
}

const response = await fetch(
API + "/predict",
{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(data)
}
)

const result = await response.json()

document.getElementById("result").innerText =
"Predicted Accident Severity: " + result.severity

if(result.severity >=3){
alert("⚠ High accident risk detected in this area");
}

}