import React, { useEffect, useState } from "react";
import { IoCamera, IoVideocam } from "react-icons/io5";

import Trending from "../../Components/NewsFeedComponent/Trending";
import Watch from "../../Components/NewsFeedComponent/Watch";
import Following from "../../Components/NewsFeedComponent/Following";
import SideTable from "../../Components/JobPostcomp/SideTable";
import BidsModal from "../../Components/BidComp/BidsModal";
import {
  getVideosApi,
  getFollowingVideosApi,
  getFeedData,
} from "../../api/Api";
import Loader from "../../Components/Loader/Loader";

const handleModal = async () => {
  document.getElementById("my_modal").showModal();
};

const NewsFeed = () => {
  const [activeTab, setactiveTab] = useState(0);
  const [videos, setVideos] = useState([]);
  const [fvideos, setFVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [channels, setChannels] = useState([]);
  const [users, setUsers] = useState([]);
  const [showContent, setshowContent] = useState(0);

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 0:
        return <Trending videos={videos} />;
      case 1:
        return <Following videos={fvideos} />;
      case 2:
        return <Watch videos={videos} />;
    }
  };

  const fetchData = async () => {
    setLoading(true);
    const data1 = await getVideosApi();
    const data2 = await getFollowingVideosApi();
    const data3 = await getFeedData();
    setUsers(data3.users);
    setChannels(data3.channels);
    setVideos(data1);
    setFVideos(data2);
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="flex w-full flex-col">
      {/*Upper  switching Tab */}
      <div className=" flex flex-wrap justify-between">
        {/*First Column */}
        <div className="w-full sm:w-[65%]">
          <div className="relative flex h-[60px] w-full items-center rounded-xl border bg-white p-4 shadow-xl">
            <div className="p-x-4 relative flex h-[40px]  w-full flex-row rounded-[10px]">
              <div
                onClick={() => setactiveTab(0)}
                className={`${
                  activeTab === 0
                    ? "w-[70%] bg-[#FF8216] text-white"
                    : "w-[70%] bg-transparent text-[#939393]"
                } flex h-full cursor-pointer items-center justify-center rounded-[10px]`}
              >
                <p className="font-bold">Trending</p>
              </div>
              <div
                onClick={() => setactiveTab(1)}
                className={`${
                  activeTab === 1
                    ? "w-[70%] bg-[#FF8216] text-white"
                    : "w-[70%] bg-transparent text-[#939393]"
                } flex h-full cursor-pointer items-center justify-center rounded-[10px] bg-[#FF8216]`}
              >
                <p className="font-bold">Following</p>
              </div>
              <div
                onClick={() => setactiveTab(2)}
                className={`${
                  activeTab === 2
                    ? "w-[70%] bg-[#FF8216] text-white"
                    : "w-[70%] bg-transparent text-[#939393]"
                } flex h-full cursor-pointer items-center justify-center rounded-[10px]  bg-[#FF8216]`}
              >
                <p className="font-bold">Watch</p>
              </div>
            </div>
          </div>

          {/* Share and Content View */}
          <div className="relative my-[22px] flex w-full flex-col items-center">
            <div className=" mb-[22px] w-[85%] rounded-[10px] bg-white shadow-xl">
              <div className="p-4 ">
                <textarea
                  placeholder="Share some news..."
                  className="textarea  textarea-sm w-full text-lg placeholder-[#f0ebeb]"
                ></textarea>

                <div className="card-actions items-center justify-between">
                  <div className="my-2 flex items-center">
                    <div className="mr-[30px] flex cursor-pointer flex-row  ">
                      <IoVideocam size={24} color="#FF8216" className="mr-3" />
                      <p>video</p>
                    </div>
                    <div className=" flex cursor-pointer flex-row ">
                      <IoCamera size={24} color="#FF8216" className="mr-3" />
                      <p>Camera</p>
                    </div>
                  </div>

                  <div className="my-2 flex cursor-pointer items-center  ">
                    <button
                      className="h-[35px] w-[100px] justify-center rounded-[10px] bg-[#FF8216]  text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583]  "
                      style={{ transition: "0.3s" }}
                    >
                      Share
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/*get data for the Switching Tabs */}
            {renderActiveComponent()}
          </div>
        </div>

        {/* Second Column View */}
        <div className="hidden w-full sm:block sm:w-[33%]">
          <div className="flex w-full items-center justify-between">
            <h1 className="text-[20px] font-bold">Top Channels</h1>
            <p
              className="cursor-pointer font-bold text-[#FF8216]"
              onClick={() => {
                handleModal();
                setshowContent(1);
              }}
            >
              See more
            </p>
          </div>

          {/*Top Channels */}
          <SideTable data={channels} />

          <div className="flex w-full items-center justify-between">
            <h1 className="text-[20px] font-bold">Top Creators</h1>
            <p
              className="cursor-pointer font-bold text-[#FF8216]"
              onClick={() => {
                handleModal();
                setshowContent(2);
              }}
            >
              See more
            </p>
          </div>

          {/*Top Creators */}
          <SideTable data={users} />
        </div>
      </div>

      {/*Modal */}
      <BidsModal
        showmodal={handleModal}
        data={users}
        channels={channels}
        showContent={showContent}
      />
    </div>
  );
};

export default NewsFeed;
