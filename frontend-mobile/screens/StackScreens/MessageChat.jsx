import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  TextInput,
  FlatList,
  Modal,
  ScrollView,
} from "react-native";
import React, { useEffect, useState, useRef } from "react";
import Ionicons from "react-native-vector-icons/Ionicons";
import url from "../../utils/url.js";
import { Avatar } from "react-native-paper";
import img1 from "../../assets/img1.png";
import Converstion from "../../components/ChatComp/Converstion.jsx";
import Toast from "react-native-toast-message";

export default function MessageChat({ navigation, route }) {
  const { userData } = route.params;
  const [modalVisible, setModalVisible] = useState(false);
  const [conversations, setConversations] = useState([]);
  const newChatArray = Array.isArray(userData.user.following)
    ? userData.user.following
    : [];

  // get converstions of the current user
  useEffect(() => {
    const getConversations = async () => {
      try {
        const res = await fetch(`${url}/conversation/${userData.user._id}`, {
          headers: {
            "content-type": "application/json",
          },
        });
        const response = await res.json();
        setConversations(response);
      } catch (err) {
        console.log(err);
      }
    };
    getConversations();
  }, [userData.user._id]);

  //Create new Converstion from following
  const createConversation = async (e) => {
    const convoArr = [];
    var exists = false;
    for (let i = 0; i < conversations.length; i++) {
      convoArr[i] = conversations[i].members.find(
        (m) => m !== userData.user._id
      );
    }
    for (let i = 0; i < conversations.length; i++) {
      if (convoArr[i] === e.user._id) {
        exists = true;
        break;
      }
    }

    if (exists === true) {
      setModalVisible(false);
      Toast.show({
        type: "error",
        text1: "Converstion already exists",
      });
    } else {
      const convo = {
        senderId: userData.user._id,
        receiverId: e.user._id,
      };
      try {
        await fetch(`${url}/conversation`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(convo),
        });
        await getConversationNow();
        setModalVisible(false);
        Toast.show({
          type: "success",
          text1: "Converstion created",
        });
      } catch (err) {
        console.log(err);
      }
    }
  };

  const getConversationNow = async () => {
    try {
      const res = await fetch(`${url}/conversation/${userData.user._id}`, {
        headers: {
          "content-type": "application/json",
        },
      });
      const response = await res.json();
      setConversations(response);
    } catch (err) {
      console.log(err);
    }
  };

  //converstions are called here
  const renderItem = ({ item }) => (
    <Converstion
      key={item._id}
      navigation={navigation}
      conversation={item}
      currentUser={userData.user}
    />
  );

  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View style={styles.uppertab}>
        <View style={styles.innertab}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="chevron-back" color={"black"} size={30} />
          </TouchableOpacity>
          <View>
            <Text
              style={{
                fontSize: 20,
                fontWeight: "medium",
                color: "black",
              }}
            >
              Chats
            </Text>
          </View>
          <TouchableOpacity onPress={() => setModalVisible(true)}>
            <Ionicons name="add" color={"black"} size={30} />
          </TouchableOpacity>
        </View>
      </View>

      <TouchableOpacity style={styles.container}>
        <Ionicons name="search" style={styles.icon} size={20} />
        <TouchableOpacity>
          <TextInput
            style={styles.input}
            placeholder="Search"
            placeholderTextColor={"darkgray"}
          />
        </TouchableOpacity>
      </TouchableOpacity>
      <FlatList
        style={{ alignSelf: "center", width: "95%" }}
        data={conversations}
        renderItem={renderItem}
        keyExtractor={(item, index) => item._id || index.toString()}
      />

      {/*create new chat Modal */}
      <Modal animationType="fade" transparent={true} visible={modalVisible}>
        <View style={styles.modalContainer}>
          <View style={styles.modal}>
            <View style={styles.modalheader}>
              <TouchableOpacity disabled>
                <Ionicons name="close" color={"transparent"} size={30} />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Start Chat with Followings</Text>
              <TouchableOpacity onPress={() => setModalVisible(false)}>
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
                  <TouchableOpacity
                    style={styles.modallistcomp}
                    key={index}
                    onPress={() => createConversation(element)}
                  >
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
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: 70,
    backgroundColor: "white",
    marginBottom: 5,
    alignItems: "center",
    shadowColor: "#171717",
    shadowOffset: { width: -2, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 10,
  },
  innertab: {
    width: "95%",
    height: "100%",
    flexDirection: "row",
    alignItems: "center",
    marginTop: "auto",
    justifyContent: "space-between",
  },
  container: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F5F5F5",
    alignSelf: "center",
    height: 40,
    borderRadius: 50,
    width: "95%",
    marginTop: 20,
    marginBottom: 20,
  },
  icon: {
    color: "darkgray",
    margin: 8,
  },
  input: {
    flex: 1,
    fontSize: 18,
    color: "darkgray",
    padding: 10,
  },

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
    justifyContent: "space-around",
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
