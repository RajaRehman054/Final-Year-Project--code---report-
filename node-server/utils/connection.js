var mongoose = require("mongoose");
require("dotenv").config();

const connectDB = async () => {
  try {
    const { connection } = await mongoose.connect('mongodb://127.0.0.1:27017/fyp', {
      useNewUrlParser: true,
    });
    console.log(`Connected with database ${connection.host} 💚 .`);
  } catch (error) {
    console.log(error);
  }
};

module.exports = { connectDB };
