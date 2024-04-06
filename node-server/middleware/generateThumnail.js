const path = require('path');
const ffmpeg = require('fluent-ffmpeg');

exports.generateThumbnail = (req, res, next) => {
	try {
		ffmpeg(
			path.resolve(
				__dirname,
				`../files/videos/${req.video._id.toString()}.mp4`
			)
		).screenshots({
			count: 1,
			folder: path.resolve(__dirname, `../files/thumbnails/`),
			filename: `${req.video._id.toString()}.png`,
		});
		next();
	} catch (error) {
		next();
	}
};
