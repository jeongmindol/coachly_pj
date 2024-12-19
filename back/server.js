const express = require("express");
const cors = require("cors");
const PORT = 8000;
const path = require("path");
const socketIo = require("socket.io");
const bodyParser = require("body-parser");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json());
app.use(bodyParser.json());

const server = app.listen(PORT, () =>
  console.log(`Server is running on ${PORT}`)
);

app.get("/", (req, res) => {
  res.send("server");
});
