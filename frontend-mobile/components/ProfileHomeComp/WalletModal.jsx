import React from "react";
import { Modal, View, Text, TouchableOpacity, StyleSheet } from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";

const WalletModal = ({ modalVisible, onCloseModal, user }) => {
  return (
    <Modal
      animationType="fade"
      transparent={true}
      visible={modalVisible}
      onRequestClose={() => onCloseModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modal}>
          <View style={styles.modalheader}>
            <TouchableOpacity disabled>
              <Ionicons name="close" color={"transparent"} size={30} />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Current Balance</Text>
            <TouchableOpacity onPress={() => onCloseModal(false)}>
              <Ionicons name="close" color={"white"} size={30} />
            </TouchableOpacity>
          </View>
          <View
            style={[
              styles.modalheader,
              { marginVertical: 5, width: "90%", alignSelf: "center" },
            ]}
          >
            <Ionicons name="cash" color={"white"} size={30} />
            <Text style={styles.modalMessage}>{user.wallet} $</Text>
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
    padding: 20,
    borderRadius: 10,
    width: "80%",
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
    color: "white",
  },
  modalMessage: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
  },
  button: {
    backgroundColor: "#1E88E5",
    borderRadius: 10,
    padding: 10,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 16,
  },
  modalheader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
});

export default WalletModal;
