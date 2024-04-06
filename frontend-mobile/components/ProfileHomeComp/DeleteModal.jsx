import React, { useState } from "react";
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";

const DeleteModal = ({ modalVisible, onCloseModal, user }) => {
  const [loading, setLoading] = useState(false);
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("delete Video");
    setLoading(true);
    Alert.alert(
      "",
      "Your Video is deleted",
      [
        {
          text: "OK",
        },
      ],
      { cancelable: true }
    );
    onCloseModal(false);
  };

  return (
    <Modal animationType="fade" transparent={true} visible={modalVisible}>
      <View style={styles.modalContainer}>
        {loading === true ? (
          <View style={styles.modal1}>
            <Text style={{ color: "black", textAlign: "center", margin: 30 }}>
              Are you sure you want to delete Video
            </Text>
            <View
              style={{
                flexDirection: "row",
                width: "100%",
                borderTopColor: "red",
                borderTopWidth: 1,
              }}
            >
              <TouchableOpacity
                style={{
                  height: 40,
                  width: "50%",
                  borderBottomLeftRadius: 10,
                  justifyContent: "center",
                  alignItems: "center",
                  borderRightColor: "red",
                  borderRightWidth: 2,
                }}
                onPress={() => onCloseModal(false)}
              >
                <Text
                  style={{
                    fontWeight: "bold",
                    color: "red",
                    fontSize: 18,
                  }}
                >
                  Cancel
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={{
                  height: 40,
                  width: "50%",
                  borderLeftColor: "red",
                  borderLeftWidth: 2,
                  borderBottomRightRadius: 10,
                  justifyContent: "center",
                  alignItems: "center",
                }}
                onPress={handleSubmit}
              >
                <Text
                  style={{
                    fontWeight: "bold",
                    color: "red",
                    fontSize: 18,
                  }}
                >
                  Delete
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        ) : (
          <View style={styles.modal}>
            <View style={styles.modalheader}>
              <TouchableOpacity disabled>
                <Ionicons name="close" color={"transparent"} size={30} />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Delete Video</Text>
              <TouchableOpacity onPress={() => onCloseModal(false)}>
                <Ionicons name="close" color={"black"} size={30} />
              </TouchableOpacity>
            </View>
            <TouchableOpacity
              style={{
                height: 50,
                backgroundColor: "red",
                borderRadius: 10,
                justifyContent: "center",
                alignItems: "center",
                marginVertical: 10,
              }}
              onPress={() => setLoading(true)}
            >
              <Text
                style={{
                  fontWeight: "bold",
                  color: "white",
                  fontSize: 20,
                }}
              >
                Delete video
              </Text>
            </TouchableOpacity>
          </View>
        )}
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
    backgroundColor: "#fff",
    padding: 10,
    borderRadius: 10,
    width: "90%",
    maxHeight: "60%",
  },
  modal1: {
    backgroundColor: "#fff",
    borderRadius: 10,
    width: "70%",
    height: "20%",
    justifyContent: "space-between",
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "transparent",
  },
  modalheader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
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

export default DeleteModal;
