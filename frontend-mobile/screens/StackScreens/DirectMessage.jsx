import React, { useState, useEffect, useRef } from "react";
import {
  Text,
  View,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  TextInput,
  FlatList,
} from "react-native";
import { Avatar } from "react-native-paper";
import Ionicons from "react-native-vector-icons/Ionicons";
import url from "../../utils/url.js";
import img1 from "../../assets/img1.png";
import MessageOther from "../../components/ChatComp/MessageOther.jsx";
import MessageOwn from "../../components/ChatComp/MessageOwn.jsx";
import { io } from "socket.io-client";

export default function DirectMessage({ navigation, route }) {
  const { conversation, currentChatUser, currentUser } = route.params;
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [currentChat, setCurrentChat] = useState(conversation);
  const [arrivalMessage, setArrivalMessage] = useState(null);
  const socket = useRef();
  const scrollRef = useRef();
  let listViewRef;

  useEffect(() => {
    socket.current = io(url);
    socket.current.on("getMessage", (data) => {
      setArrivalMessage({
        sender: data.senderId,
        text: data.text,
        createdAt: Date.now(),
      });
    });
  }, []);

  useEffect(() => {
    arrivalMessage &&
      currentChat?.members.includes(arrivalMessage.sender) &&
      setMessages((prev) => [...prev, arrivalMessage]);
  }, [arrivalMessage, currentChat]);

  useEffect(() => {
    socket.current.emit("addUser", currentUser._id);
  }, [currentUser]);

  // get messages of a current converstion
  useEffect(() => {
    const getMessages = async () => {
      try {
        const res = await fetch(
          `${url}/conversation/getmessages/${currentChat._id}`,
          {
            headers: {
              "content-type": "application/json",
            },
          }
        );
        const response = await res.json();
        setMessages(response);
      } catch (err) {
        console.log(err);
      }
    };
    getMessages();
  }, [currentChat]);

  // create new message
  const handleSubmit = async (e) => {
    e.preventDefault();
    const message = {
      sender: currentUser._id,
      text: newMessage,
      conversationId: currentChat._id,
    };
    const receiverId = currentChat.members.find(
      (member) => member !== currentUser._id
    );

    socket.current.emit("sendMessage", {
      senderId: currentUser._id,
      receiverId,
      text: newMessage,
    });

    try {
      const res = await fetch(`${url}/conversation/newmessage`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(message),
      });
      const response = await res.json();
      setMessages([...messages, response]);
      setNewMessage("");
    } catch (err) {
      console.log(err);
    }
  };

  useEffect(() => {
    if (messages.length > 0) {
      listViewRef.scrollToEnd({ animated: true });
    }
  }, [messages]);

  //messages are called here
  const renderItem = ({ item }) =>
    item.sender === currentUser._id ? (
      <MessageOwn item={item} />
    ) : (
      <MessageOther item={item} />
    );

  return (
    <SafeAreaView style={{ backgroundColor: "white", flex: 1 }}>
      <View>
        {/*top bar showing profile info */}
        <View style={styles.uppertab}>
          <View style={styles.innertab}>
            <TouchableOpacity onPress={() => navigation.goBack()}>
              <Ionicons name="chevron-back" color={"black"} size={30} />
            </TouchableOpacity>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
              }}
            >
              <Avatar.Image
                source={
                  currentChatUser.picture
                    ? {
                        uri: `${url}/users/picture?path=${currentChatUser.picture}`,
                      }
                    : img1
                }
                size={45}
                style={{ marginRight: 10 }}
              />
              <Text
                style={{
                  fontSize: 20,
                  fontWeight: "bold",
                  color: "black",
                  marginLeft: 10,
                }}
              >
                {currentChatUser.username}
              </Text>
            </View>
            <TouchableOpacity>
              <Ionicons name="ellipsis-horizontal" color={"black"} size={30} />
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/*chat area showing  messages */}
      {messages.length === 0 ? (
        <View
          style={{
            width: "100%",
            flex: 1,
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Text style={{ color: "black", fontSize: 16, fontWeight: "bold" }}>
            Be the First to send a message
          </Text>
        </View>
      ) : (
        <FlatList
          ref={(ref) => {
            listViewRef = ref;
          }}
          data={messages}
          renderItem={renderItem}
          keyExtractor={(item, index) =>
            item.id ? item.id.toString() : index.toString()
          }
          style={{
            width: "100%",
            flex: 1,
          }}
        />
      )}

      {/*type messages area */}
      <View style={styles.buttonview}>
        <View style={styles.messagebar}>
          <View style={styles.container}>
            <TextInput
              style={styles.input}
              placeholder="Type message...."
              placeholderTextColor={"darkgray"}
              onChangeText={setNewMessage}
              value={newMessage}
            />
          </View>
          <TouchableOpacity
            style={styles.iconview}
            onPress={handleSubmit}
            disabled={!newMessage.trim()}
          >
            <Ionicons name="paper-plane" color={"white"} size={25} />
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  uppertab: {
    height: 70,
    backgroundColor: "white",
    marginBottom: 10,
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
  buttonview: {
    flexDirection: "row",
    height: 80,
    width: "90%",
    alignSelf: "center",
    justifyContent: "space-between",
    alignItems: "center",
  },
  messagebar: {
    width: "100%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  iconview: {
    width: "16%",
    height: 50,
    borderRadius: 50,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FF8216",
    elevation: 5,
  },
  container: {
    width: "82%",
    padding: 4,
    backgroundColor: "#F5F5F5",
    elevation: 5,
    height: 50,
    borderRadius: 50,
  },
  input: {
    fontSize: 18,
    color: "darkgray",
    padding: 10,
  },
});
