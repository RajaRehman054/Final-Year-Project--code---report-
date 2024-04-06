import React, { useState, useEffect, useContext } from "react";
import {
  FlatList,
  StyleSheet,
  Image,
  TouchableOpacity,
  View,
  Text,
} from "react-native";
import { AuthContext } from "../../auth/AuthContext";
import Loader from "../Loader";
import ColumnVideoModal from "./ColumnVideoModal";

export default function ColumnView() {
  const [loading, setLoading] = useState(false);
  const { user } = useContext(AuthContext);
  const [modalVisible, setModalVisible] = useState(false);
  const [data, setData] = useState([]);
  const getVideos = async () => {
    setLoading(true);
    const res = await fetch(`${url}/videos/myvideos`, {
      headers: {
        "content-type": "application/json",
        Authorization: `Bearer ${user}`,
      },
    });
    const response = await res.json();
    setData(response);
    setLoading(false);
  };

  useEffect(() => {
    getVideos();
  }, []);

  const onCloseModal = (visible) => {
    setModalVisible(visible);
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity style={styles.item} onPress={() => setModalVisible(true)}>
      <Image
        source={{
          uri: `${url}/videos/thumbnails?path=${item.thumbnail}`,
        }}
        style={styles.image}
      />
      <ColumnVideoModal
        modalVisible={modalVisible}
        onCloseModal={onCloseModal}
        video={item}
      />
    </TouchableOpacity>
  );

  if (loading) {
    return <Loader />;
  }

  return (
    <>
      {data.length === 0 ? (
        <View
          style={{
            alignSelf: "center",
            height: 200,
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Text
            style={{
              color: "#FF8216",
              textAlign: "center",
              fontWeight: "bold",
              fontSize: 20,
            }}
          >
            No videos to Show
          </Text>
        </View>
      ) : (
        <FlatList
          data={data}
          keyExtractor={(item) => item._id}
          numColumns={2}
          renderItem={renderItem}
          style={{ width: "95%", alignSelf: "center" }}
        />
      )}
    </>
  );
}

const styles = StyleSheet.create({
  item: {
    flex: 1,
    height: 100,
    margin: 5,
    borderRadius: 10,
  },
  image: {
    width: "100%",
    height: "100%",
    borderRadius: 10,
  },
  screen: { backgroundColor: "#d9d9d9", flex: 1 },
});
