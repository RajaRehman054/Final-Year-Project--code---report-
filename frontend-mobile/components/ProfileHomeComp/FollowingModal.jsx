import React from "react";
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from "react-native";
import Ionicons from "react-native-vector-icons/Ionicons";
import { Avatar } from "react-native-paper";
import img1 from "../../assets/img1.png";

const FollowingModal = ({ modalVisible, onCloseModal, user }) => {
  const newChatArray = Array.isArray(user.following) ? user.following : [];

  return (
    <Modal animationType="fade" transparent={true} visible={modalVisible}>
      <View style={styles.modalContainer}>
        <View style={styles.modal}>
          <View style={styles.modalheader}>
            <TouchableOpacity disabled>
              <Ionicons name="close" color={"transparent"} size={30} />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Following</Text>
            <TouchableOpacity onPress={() => onCloseModal(false)}>
              <Ionicons name="close" color={"black"} size={30} />
            </TouchableOpacity>
          </View>
          {newChatArray.length === 0 ? (
            <TouchableOpacity style={styles.modallistcomp} disabled>
              <Text
                style={{
                  fontWeight: "bold",
                  color: "black",
                }}
              >
                0 Following
              </Text>
            </TouchableOpacity>
          ) : (
            <ScrollView>
              {newChatArray.map((element, index) => (
                <TouchableOpacity style={styles.modallistcomp} key={index}>
                  <View style={styles.iconimg}>
                    <Avatar.Image
                      source={
                        element.user.picture
                          ? {
                              uri: `${url}/users/picture?path=${element.user.picture}`,
                            }
                          : img1
                      }
                      size={50}
                    />
                  </View>
                  <Text
                    style={{
                      fontWeight: "bold",
                      color: "black",
                      marginLeft: 10,
                    }}
                  >
                    {element.user.username}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          )}
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
    backgroundColor: "#fff",
    padding: 10,
    borderRadius: 10,
    width: "90%",
    maxHeight: "60%",
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#FF8216",
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

export default FollowingModal;
