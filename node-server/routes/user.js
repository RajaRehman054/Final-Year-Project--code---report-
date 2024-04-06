var express = require("express");
var router = express.Router();
var passport = require("passport");

var authenticate = require("../middleware/auth");
var profilingModule = require("../controllers/profilingModule");
var pictureHandler = require("../middleware/pictureHandler");
var User = require("../models/user");

// ? Profiling Module Routes //
router.get("/wallet/inc/:user/:amount", profilingModule.updateWalletInc);
router.get("/otp/:email", profilingModule.getOtp);
router.get("/otpVerify/:email/:otp", profilingModule.verifyOtp);
router.get("/picture", profilingModule.getPicture);
router.get("/user", authenticate.verifyUser, profilingModule.getUser);
router.post("/register", profilingModule.register);
router.post("/login", passport.authenticate("local"), profilingModule.signIn);
router.patch("/fcm", authenticate.verifyUser, profilingModule.setFCM);
router.patch("/passwordreset", profilingModule.passwordReset);
router.patch(
  "/profilepicture",
  authenticate.verifyUser,
  pictureHandler.uploadPicture.single("picture"),
  profilingModule.profilePicture
);
router.patch(
  "/profilepicture/web",
  authenticate.verifyUser,
  profilingModule.profilePictureWeb
);
router.patch(
  "/profile/edit",
  authenticate.verifyUser,
  profilingModule.editProfile
);
router.patch(
  "/profile/password/change",
  passport.authenticate("local"),
  authenticate.verifyUser,
  profilingModule.passwordChange
);
router.patch(
  "/follower/add/:id",
  authenticate.verifyUser,
  profilingModule.addFollower
);
router.patch(
  "/follower/remove/:id",
  authenticate.verifyUser,
  profilingModule.removeFollower
);
router.patch(
  "/wallet/dec",
  authenticate.verifyUser,
  profilingModule.updateWalletDec
);
router.patch(
  "/job/request/:id",
  authenticate.verifyUser,
  profilingModule.addRequests
);

//TODO: get a user
router.get("/", profilingModule.getOtherUser);

//TODO: postings
router.get("/jobs", authenticate.verifyUser, profilingModule.getCurrentJobs);
router.get(
  "/jobs/applied",
  authenticate.verifyUser,
  profilingModule.getAppliedJobs
);
router.get(
  "/jobs/hiredJob",
  authenticate.verifyUser,
  profilingModule.getCurrentHiredJob
);
module.exports = router;
