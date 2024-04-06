import { Modal, View, Text, TouchableOpacity, StyleSheet } from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";
import VideoPlayer from "../VideoPlayerComp/VideoPlayer";
const ColumnVideoModal = ({ modalVisible, onCloseModal, video }) => {
  return (
    <Modal animationType="fade" transparent={true} visible={modalVisible}>
      <View style={styles.modalContainer}>
        <View style={styles.modal}>
          <View style={styles.modalheader}>
            <TouchableOpacity disabled>
              <Ionicons name="close" color={"transparent"} size={30} />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Video</Text>
            <TouchableOpacity onPress={() => onCloseModal(false)}>
              <Ionicons name="close" color={"white"} size={30} />
            </TouchableOpacity>
          </View>
          <View
            style={{
              width: "100%",
              height: 200,
              alignSelf: "center",
            }}
          >
            <VideoPlayer fullscreenIcon={false} source={video.path} />
          </View>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  modal: {
    backgroundColor: "#FF8216",
    padding: 10,
    borderRadius: 10,
    width: "90%",
    maxHeight: "60%",
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
  },
  modalheader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 5,
  },
  modalMessage: {
    fontSize: 16,
    color: "black",
  },
  modallistcomp: {
    height: 60,
    flexDirection: "row",
    padding: 5,
    alignItems: "center",
  },
});

export default ColumnVideoModal;
