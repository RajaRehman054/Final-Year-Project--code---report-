const multer = require('multer');
const path = require('path');
const fs = require('fs');

const storage = multer.diskStorage({
	destination: (req, file, cb) => {
		cb(null, path.resolve(__dirname, '../files/pictures'));
	},
	filename: async (req, file, cb) => {
		if (req.user.picture) {
			fs.unlink(path.resolve(__dirname, `..${req.user.picture}`), err => {
				if (err) {
					console.log(err);
				}
			});
		}
		const uniqueName = `${Date.now()}-${Math.round(
			Math.random() * 1e9
		)}${path.extname(file.originalname)}.jpg`;
		cb(null, uniqueName);
		req.source = `/files/pictures/${uniqueName}`;
	},
});

exports.uploadPicture = multer({ storage: storage });
